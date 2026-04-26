"""Train LightGBM-based inverse model: (ec, ph) -> (nutrient_ml, ph_up_ml, ph_down_ml).

One LGBMRegressor per output channel — the three dose channels are mostly
independent (nutrient ↔ EC, pH adjusters ↔ pH; cross-coupling is small),
so independent regressors with per-channel early stopping is a clean fit.
"""

from __future__ import annotations
from pathlib import Path
import json

import joblib
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from data_loader import load_dataset, split_train_val


# Action bounds (see project_hydroponic_dosing.md): clipped post-prediction.
DOSE_CAPS: dict[str, float] = {
    "nutrient_ml": 2.0,
    "ph_up_ml": 0.5,
    "ph_down_ml": 0.5,
}

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=10,
    objective="regression",
    random_state=0,
    verbose=-1,
)


def train_one(
    X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> LGBMRegressor:
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=0)],
    )
    return model


def predict_clipped(
    models: dict[str, LGBMRegressor], X: np.ndarray, target_names: tuple[str, ...]
) -> np.ndarray:
    preds = np.column_stack([models[t].predict(X) for t in target_names])
    for j, t in enumerate(target_names):
        preds[:, j] = np.clip(preds[:, j], 0.0, DOSE_CAPS[t])
    return preds


def per_channel_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for j, n in enumerate(names):
        diff = y_pred[:, j] - y_true[:, j]
        ss_res = float(np.sum(diff**2))
        ss_tot = float(np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2))
        out[n] = {
            "mse": float(np.mean(diff**2)),
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
    print(f"Train/val split (by session): " f"train={len(train.X)}  val={len(val.X)}")

    models: dict[str, LGBMRegressor] = {}
    for j, name in enumerate(training_pool.target_names):
        print(f"\nTraining {name}...")
        models[name] = train_one(train.X, train.y[:, j], val.X, val.y[:, j])
        best_iter = models[name].best_iteration_
        print(f"  best_iteration={best_iter}")

    metrics = {}
    for split_name, ds in [("val", val), ("test", test)]:
        preds = predict_clipped(models, ds.X, training_pool.target_names)
        m = per_channel_metrics(ds.y, preds, training_pool.target_names)
        metrics[split_name] = m
        print()
        print(_format_metrics(split_name, m))

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "feature_names": training_pool.feature_names,
            "target_names": training_pool.target_names,
            "dose_caps": DOSE_CAPS,
            "lgbm_params": LGBM_PARAMS,
        },
        out_dir / "lgbm_inverse.joblib",
    )
    with (out_dir / "lgbm_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model -> {out_dir/'lgbm_inverse.joblib'}")
    print(f"Saved metrics -> {out_dir/'lgbm_metrics.json'}")


if __name__ == "__main__":
    main()
