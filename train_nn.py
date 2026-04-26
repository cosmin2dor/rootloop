"""Train an MLP-based inverse model: (ec, ph) -> (nutrient_ml, ph_up_ml, ph_down_ml).

Drop-in counterpart to train_lgbm.py — same data loader, same train/val/test
split, same dose caps, same metric format. Saves to models/nn_inverse.pt.
"""

from __future__ import annotations
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_dataset, split_train_val
from nn_model import DoseNet


# Action bounds (see project_hydroponic_dosing.md): clipped post-prediction.
DOSE_CAPS: dict[str, float] = {
    "nutrient_ml": 2.0,
    "ph_up_ml": 0.5,
    "ph_down_ml": 0.5,
}

NN_PARAMS = dict(
    hidden_dim=32,
    learning_rate=1e-3,
    batch_size=64,
    max_epochs=200,
    patience=20,
    weight_decay=0.0,
    seed=0,
)


def standardize_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    max_epochs: int,
    patience: int,
    weight_decay: float,
) -> tuple[nn.Module, float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total = 0.0
            n = 0
            for X_b, y_b in val_loader:
                total += loss_fn(model(X_b), y_b).item() * len(X_b)
                n += len(X_b)
            val_loss = total / n

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {epoch} (best {best_epoch}, val {best_val:.6f})")
                break
    else:
        print(f"  reached max_epochs={max_epochs} (best {best_epoch}, val {best_val:.6f})")

    model.load_state_dict(best_state)
    return model, best_val, best_epoch


def predict_clipped(
    model: nn.Module,
    X_norm: np.ndarray,
    target_names: tuple[str, ...],
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_norm).float()).numpy()
    for j, t in enumerate(target_names):
        preds[:, j] = np.clip(preds[:, j], 0.0, DOSE_CAPS[t])
    return preds


def per_channel_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for j, n in enumerate(names):
        diff = y_pred[:, j] - y_true[:, j]
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2))
        out[n] = {
            "mse": float(np.mean(diff ** 2)),
            "mae": float(np.mean(np.abs(diff))),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        }
    return out


def _format_metrics(name: str, m: dict[str, dict[str, float]]) -> str:
    lines = [f"{name} metrics:"]
    for n, vals in m.items():
        lines.append(
            f"  {n:>12}: mse={vals['mse']:.5f}  mae={vals['mae']:.4f}  r2={vals['r2']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    torch.manual_seed(NN_PARAMS["seed"])
    np.random.seed(NN_PARAMS["seed"])

    training_pool = load_dataset("data/training")
    test = load_dataset("data/test")
    print(
        f"Loaded training pool: {len(training_pool.X)} samples "
        f"from {len(np.unique(training_pool.session_ids))} sessions"
    )
    print(
        f"Loaded test set:      {len(test.X)} samples "
        f"from {len(np.unique(test.session_ids))} sessions"
    )

    train, val = split_train_val(training_pool, val_frac=0.2, seed=0)
    print(f"Train/val split (by session): train={len(train.X)}  val={len(val.X)}")

    mean, std = standardize_stats(train.X)
    print(f"Feature standardization: mean={mean.tolist()}, std={std.tolist()}")

    X_train = (train.X - mean) / std
    X_val = (val.X - mean) / std
    X_test = (test.X - mean) / std

    bs = NN_PARAMS["batch_size"]
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(train.y).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(val.y).float())
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    model = DoseNet(
        input_dim=X_train.shape[1],
        hidden_dim=NN_PARAMS["hidden_dim"],
        output_dim=train.y.shape[1],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining DoseNet ({n_params} parameters)...")
    model, best_val, best_epoch = _train_loop(
        model,
        train_loader,
        val_loader,
        lr=NN_PARAMS["learning_rate"],
        max_epochs=NN_PARAMS["max_epochs"],
        patience=NN_PARAMS["patience"],
        weight_decay=NN_PARAMS["weight_decay"],
    )

    metrics = {}
    for split_name, ds, X_norm in [("val", val, X_val), ("test", test, X_test)]:
        preds = predict_clipped(model, X_norm, training_pool.target_names)
        m = per_channel_metrics(ds.y, preds, training_pool.target_names)
        metrics[split_name] = m
        print()
        print(_format_metrics(split_name, m))

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": {
                "input_dim": int(X_train.shape[1]),
                "hidden_dim": NN_PARAMS["hidden_dim"],
                "output_dim": int(train.y.shape[1]),
            },
            "feature_names": list(training_pool.feature_names),
            "target_names": list(training_pool.target_names),
            "input_mean": mean.tolist(),
            "input_std": std.tolist(),
            "dose_caps": DOSE_CAPS,
            "nn_params": NN_PARAMS,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        out_dir / "nn_inverse.pt",
    )
    with (out_dir / "nn_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model -> {out_dir / 'nn_inverse.pt'}")
    print(f"Saved metrics -> {out_dir / 'nn_metrics.json'}")


if __name__ == "__main__":
    main()
