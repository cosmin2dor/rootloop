"""Load simulated session events and build (state, action) training pairs.

Each `read` event becomes one row. The action label is the dose volumes from
a co-timestamped `dose` event if present, otherwise (0, 0, 0). Train/val/test
splits are by session UUID — never split within a session.
"""

from __future__ import annotations
from pathlib import Path
from typing import NamedTuple
import json

import numpy as np


FEATURE_COLS: tuple[str, ...] = ("ec_observed", "ph_observed")
TARGET_COLS: tuple[str, ...] = ("nutrient_ml", "ph_up_ml", "ph_down_ml")


class Dataset(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    session_ids: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


def _session_to_rows(events: list[dict]) -> list[dict]:
    doses_by_ts = {e["ts"]: e for e in events if e["event_type"] == "dose"}
    rows = []
    for e in events:
        if e["event_type"] != "read":
            continue
        dose = doses_by_ts.get(e["ts"])
        rows.append(
            {
                "ec_observed": e["ec_observed"],
                "ph_observed": e["ph_observed"],
                "nutrient_ml": dose["nutrient_ml"] if dose else 0.0,
                "ph_up_ml": dose["ph_up_ml"] if dose else 0.0,
                "ph_down_ml": dose["ph_down_ml"] if dose else 0.0,
            }
        )
    return rows


def load_dataset(data_dir: str | Path = "data") -> Dataset:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No session files found in {data_dir}/")

    rows: list[dict] = []
    session_ids: list[str] = []
    for path in files:
        with path.open() as f:
            events = json.load(f)
        for r in _session_to_rows(events):
            rows.append(r)
            session_ids.append(path.stem)

    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y = np.array([[r[c] for c in TARGET_COLS] for r in rows], dtype=np.float32)
    return Dataset(
        X=X,
        y=y,
        session_ids=np.array(session_ids),
        feature_names=FEATURE_COLS,
        target_names=TARGET_COLS,
    )


def split_train_val(
    dataset: Dataset,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train/val by session UUID. Test set is loaded separately."""
    sessions = np.unique(dataset.session_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(sessions)

    n_val = max(1, int(len(sessions) * val_frac))
    if n_val >= len(sessions):
        raise ValueError(f"val_frac={val_frac} leaves no training sessions ({len(sessions)} total)")
    val_ids = set(sessions[:n_val])
    train_ids = set(sessions[n_val:])

    def _slice(ids: set) -> Dataset:
        mask = np.array([s in ids for s in dataset.session_ids])
        return Dataset(
            X=dataset.X[mask],
            y=dataset.y[mask],
            session_ids=dataset.session_ids[mask],
            feature_names=dataset.feature_names,
            target_names=dataset.target_names,
        )

    return _slice(train_ids), _slice(val_ids)
