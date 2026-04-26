"""Run hydroponic dosing sessions, or compare controllers in closed-loop.

Two subcommands:

  generate   Run sessions and save events to JSON files.
             python run_session.py generate --controller rule --num_sessions 50 \\
                 --out_dir data/training

  compare    Run all listed controllers on matched seeds (same init + noise per pair),
             print a side-by-side metric table.
             python run_session.py compare --controllers rule,lgbm --num_sessions 50

Events conform to schemas/events.schema.json (minus schema_version).
"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, pstdev
import argparse
import json
import random
import uuid

from simulator import HydroponicSim, SimConfig


# Sampling ranges for randomized initial conditions.
INITIAL_EC_RANGE = (0.8, 2.5)
INITIAL_PH_RANGE = (4.5, 7.5)

# Predictions below this volume (ml) are treated as zero — pump precision floor.
MIN_DOSE_ML = 0.05

# Comparison thresholds (project_hydroponic_dosing.md).
TRANSIENT_SKIP_CYCLES = 20
PH_ALARM_LOW, PH_ALARM_HIGH = 4.5, 7.5
EC_ALARM_LOW, EC_ALARM_HIGH = 0.5, 4.0
PH_SAFE_LOW, PH_SAFE_HIGH = 5.0, 7.0
EC_SAFE_LOW, EC_SAFE_HIGH = 1.0, 3.0


@dataclass
class Setpoints:
    ph_setpoint: float = 6.0
    ph_deadband_low: float = 5.7
    ph_deadband_high: float = 6.3
    ec_setpoint: float = 1.8
    ec_deadband_low: float = 1.6
    ec_deadband_high: float = 2.0


Controller = Callable[[float, float, "Setpoints"], tuple[float, float, float]]


# ---------------------------------------------------------------- event/session


def make_event(
    ts: datetime,
    event_type: str,
    ec: float,
    ph: float,
    volume_l: float,
    nutrient_ml: float,
    ph_up_ml: float,
    ph_down_ml: float,
    sp: Setpoints,
) -> dict:
    return {
        "ts": ts.isoformat(),
        "event_type": event_type,
        "ec_observed": round(ec, 4),
        "ph_observed": round(ph, 4),
        "volume_l": round(volume_l, 4),
        "nutrient_ml": round(nutrient_ml, 4),
        "ph_up_ml": round(ph_up_ml, 4),
        "ph_down_ml": round(ph_down_ml, 4),
        "ph_setpoint": sp.ph_setpoint,
        "ph_deadband_low": sp.ph_deadband_low,
        "ph_deadband_high": sp.ph_deadband_high,
        "ec_setpoint": sp.ec_setpoint,
        "ec_deadband_low": sp.ec_deadband_low,
        "ec_deadband_high": sp.ec_deadband_high,
    }


def run_session(
    duration_h: float = 6.0,
    wait_s: float = 120.0,
    sim_dt_s: float = 1.0,
    seed: int = 0,
    config: SimConfig | None = None,
    controller: Controller | None = None,
) -> list[dict]:
    if controller is None:
        controller = rule_based_dose
    sim = HydroponicSim(config=config, seed=seed)
    sp = Setpoints()
    events: list[dict] = []

    t = datetime(2026, 4, 26, 0, 0, 0, tzinfo=timezone.utc)
    end = t + timedelta(hours=duration_h)

    while t < end:
        ec, ph = sim.read()
        events.append(make_event(t, "read", ec, ph, sim.V, 0.0, 0.0, 0.0, sp))

        n_ml, up_ml, dn_ml = controller(ec, ph, sp)
        if n_ml > 0 or up_ml > 0 or dn_ml > 0:
            events.append(make_event(t, "dose", ec, ph, sim.V, n_ml, up_ml, dn_ml, sp))
            sim.dose(n_ml, up_ml, dn_ml)

        steps = int(wait_s / sim_dt_s)
        for _ in range(steps):
            sim.step(sim_dt_s)
        t += timedelta(seconds=wait_s)

    return events


# ---------------------------------------------------------------- controllers


def rule_based_dose(ec: float, ph: float, sp: Setpoints) -> tuple[float, float, float]:
    """Stand-in controller for generating training data."""
    nutrient_ml = ph_up_ml = ph_down_ml = 0.0

    if ec < sp.ec_deadband_low:
        gap = sp.ec_setpoint - ec
        nutrient_ml = round(min(gap * 1.5, 2.0), 2)

    if ph > sp.ph_deadband_high:
        ph_down_ml = round(min((ph - sp.ph_setpoint) * 0.3, 0.5), 2)
    elif ph < sp.ph_deadband_low:
        ph_up_ml = round(min((sp.ph_setpoint - ph) * 0.3, 0.5), 2)

    return nutrient_ml, ph_up_ml, ph_down_ml


def make_lgbm_controller(model_path: str = "models/lgbm_inverse.joblib") -> Controller:
    """Load the LGBM model bundle and return a controller fn matching rule_based_dose."""
    import joblib
    import numpy as np

    bundle = joblib.load(model_path)
    models = bundle["models"]
    target_names = tuple(bundle["target_names"])
    dose_caps = bundle["dose_caps"]

    expected = ("nutrient_ml", "ph_up_ml", "ph_down_ml")
    if target_names != expected:
        raise ValueError(
            f"Unexpected target order in {model_path}: {target_names}, expected {expected}"
        )

    def predict(ec: float, ph: float, _sp: Setpoints) -> tuple[float, float, float]:
        X = np.array([[ec, ph]], dtype=np.float32)
        out: list[float] = []
        for t in target_names:
            v = float(models[t].predict(X)[0])
            v = max(0.0, min(v, dose_caps[t]))
            if v < MIN_DOSE_ML:
                v = 0.0
            out.append(round(v, 4))
        return out[0], out[1], out[2]

    return predict


def make_nn_controller(model_path: str = "models/nn_inverse.pt") -> Controller:
    """Load the NN model bundle and return a controller fn matching rule_based_dose."""
    import numpy as np
    import torch
    from nn_model import DoseNet

    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    arch = bundle["architecture"]
    target_names = tuple(bundle["target_names"])
    dose_caps = bundle["dose_caps"]
    input_mean = np.array(bundle["input_mean"], dtype=np.float32)
    input_std = np.array(bundle["input_std"], dtype=np.float32)

    expected = ("nutrient_ml", "ph_up_ml", "ph_down_ml")
    if target_names != expected:
        raise ValueError(
            f"Unexpected target order in {model_path}: {target_names}, expected {expected}"
        )

    model = DoseNet(**arch)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    def predict(ec: float, ph: float, _sp: Setpoints) -> tuple[float, float, float]:
        X = (np.array([[ec, ph]], dtype=np.float32) - input_mean) / input_std
        with torch.no_grad():
            preds = model(torch.from_numpy(X)).numpy()[0]
        out: list[float] = []
        for j, t in enumerate(target_names):
            v = float(preds[j])
            v = max(0.0, min(v, dose_caps[t]))
            if v < MIN_DOSE_ML:
                v = 0.0
            out.append(round(v, 4))
        return out[0], out[1], out[2]

    return predict


# Registry of available controllers — extend here when adding new ones.
AVAILABLE_CONTROLLERS = ["rule", "lgbm", "nn"]


def get_controller(name: str, args: argparse.Namespace) -> Controller:
    if name == "rule":
        return rule_based_dose
    if name == "lgbm":
        return make_lgbm_controller(args.lgbm_model)
    if name == "nn":
        return make_nn_controller(args.nn_model)
    raise ValueError(f"Unknown controller {name!r}; available: {AVAILABLE_CONTROLLERS}")


# ---------------------------------------------------------------- comparison metrics


def _first_in_band(values: list[float], low: float, high: float) -> int | None:
    for i, v in enumerate(values):
        if low <= v <= high:
            return i
    return None


def _frac_outside(values: list[float], low: float, high: float) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v < low or v > high) / len(values)


def session_metrics(events: list[dict], sp: Setpoints) -> dict:
    reads = [e for e in events if e["event_type"] == "read"]
    doses = [e for e in events if e["event_type"] == "dose"]
    ph_vals = [e["ph_observed"] for e in reads]
    ec_vals = [e["ec_observed"] for e in reads]
    tail_ph = ph_vals[TRANSIENT_SKIP_CYCLES:]
    tail_ec = ec_vals[TRANSIENT_SKIP_CYCLES:]
    return {
        "ttd_ph": _first_in_band(ph_vals, sp.ph_deadband_low, sp.ph_deadband_high),
        "ttd_ec": _first_in_band(ec_vals, sp.ec_deadband_low, sp.ec_deadband_high),
        "oosz_ph_pct": 100 * _frac_outside(ph_vals, PH_SAFE_LOW, PH_SAFE_HIGH),
        "oosz_ec_pct": 100 * _frac_outside(ec_vals, EC_SAFE_LOW, EC_SAFE_HIGH),
        "ss_std_ph": pstdev(tail_ph) if len(tail_ph) >= 2 else float("nan"),
        "ss_std_ec": pstdev(tail_ec) if len(tail_ec) >= 2 else float("nan"),
        "alarm_ph": any(v < PH_ALARM_LOW or v > PH_ALARM_HIGH for v in ph_vals),
        "alarm_ec": any(v < EC_ALARM_LOW or v > EC_ALARM_HIGH for v in ec_vals),
        "total_nutrient_ml": sum(d["nutrient_ml"] for d in doses),
        "total_ph_up_ml": sum(d["ph_up_ml"] for d in doses),
        "total_ph_down_ml": sum(d["ph_down_ml"] for d in doses),
        "n_doses": len(doses),
    }


def _agg(metrics_list: list[dict], key: str) -> tuple[float, float, int]:
    """Return (mean, std, n_valid). Booleans count as 0/1; None values are skipped."""
    vals: list[float] = []
    for m in metrics_list:
        v = m[key]
        if v is None:
            continue
        if isinstance(v, bool):
            vals.append(1.0 if v else 0.0)
        else:
            vals.append(float(v))
    if not vals:
        return float("nan"), float("nan"), 0
    return mean(vals), (pstdev(vals) if len(vals) > 1 else 0.0), len(vals)


METRIC_LAYOUT = [
    ("ttd_ph", "cycles to pH deadband"),
    ("ttd_ec", "cycles to EC deadband"),
    ("oosz_ph_pct", "% time pH outside safe zone"),
    ("oosz_ec_pct", "% time EC outside safe zone"),
    ("ss_std_ph", "steady-state pH std"),
    ("ss_std_ec", "steady-state EC std"),
    ("alarm_ph", "fraction sessions w/ pH alarm"),
    ("alarm_ec", "fraction sessions w/ EC alarm"),
    ("total_nutrient_ml", "total nutrient ml/session"),
    ("total_ph_up_ml", "total pH-up ml/session"),
    ("total_ph_down_ml", "total pH-down ml/session"),
    ("n_doses", "doses per session"),
]


# ---------------------------------------------------------------- subcommands


def cmd_generate(args: argparse.Namespace) -> None:
    controller = get_controller(args.controller, args)
    print(f"Controller: {args.controller}")
    print(f"Output dir: {args.out_dir}")

    for _ in range(args.num_sessions):
        session_uuid = uuid.uuid4()
        seed = session_uuid.int & 0xFFFFFFFF
        init_rng = random.Random(seed)
        initial_ec = init_rng.uniform(*INITIAL_EC_RANGE)
        initial_ph = init_rng.uniform(*INITIAL_PH_RANGE)
        config = SimConfig(initial_ec=initial_ec, initial_ph=initial_ph)

        events = run_session(
            duration_h=args.duration_h, seed=seed, config=config, controller=controller
        )
        out_path = Path(args.out_dir) / f"{session_uuid}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(events, f, indent=2)
        n_dose = sum(1 for e in events if e["event_type"] == "dose")
        n_read = sum(1 for e in events if e["event_type"] == "read")
        print(f"Session {session_uuid}")
        print(f"  initial EC={initial_ec:.2f} mS/cm, pH={initial_ph:.2f}")
        print(f"  {len(events)} events ({n_read} read, {n_dose} dose) -> {out_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    names = [n.strip() for n in args.controllers.split(",") if n.strip()]
    for name in names:
        if name not in AVAILABLE_CONTROLLERS:
            raise SystemExit(
                f"Unknown controller {name!r}; available: {AVAILABLE_CONTROLLERS}"
            )
    print(f"Comparing controllers: {names}")
    controllers = {name: get_controller(name, args) for name in names}
    sp = Setpoints()

    master = random.Random(args.seed)
    pair_seeds = [master.randint(0, 0xFFFFFFFF) for _ in range(args.num_sessions)]

    results: dict[str, list[dict]] = {name: [] for name in names}
    for i, seed in enumerate(pair_seeds):
        rng = random.Random(seed)
        initial_ec = rng.uniform(*INITIAL_EC_RANGE)
        initial_ph = rng.uniform(*INITIAL_PH_RANGE)
        config = SimConfig(initial_ec=initial_ec, initial_ph=initial_ph)
        for name, ctrl in controllers.items():
            events = run_session(
                duration_h=args.duration_h, seed=seed, config=config, controller=ctrl
            )
            results[name].append(session_metrics(events, sp))
        if (i + 1) % 10 == 0 or (i + 1) == args.num_sessions:
            print(f"  pair {i + 1}/{args.num_sessions}")

    # Print side-by-side table.
    label_width = 32
    col_width = 19  # exactly fits "  m.mmm ± s.sss"
    line_width = label_width + col_width * len(names)

    print()
    print("=" * line_width)
    print(f"Comparison over {args.num_sessions} matched pairs (same seed -> same init + noise)")
    print("=" * line_width)
    header = f"{'metric':<{label_width}s}"
    for name in names:
        header += f"  {name:>{col_width - 2}s}"
    print(header)
    print("-" * line_width)

    summary: dict[str, dict] = {}
    for key, label in METRIC_LAYOUT:
        line = f"{label:<{label_width}s}"
        col_data: dict[str, dict] = {}
        for name in names:
            m, s, n = _agg(results[name], key)
            line += f"  {m:>7.3f} ± {s:<7.3f}"
            col_data[name] = {"mean": m, "std": s, "n_valid": n}
        print(line)
        summary[key] = col_data
    print("=" * line_width)

    # Failure-to-settle summary (ttd is None for sessions that never reach deadband).
    for name in names:
        no_ph = sum(1 for m in results[name] if m["ttd_ph"] is None)
        no_ec = sum(1 for m in results[name] if m["ttd_ec"] is None)
        print(
            f"  {name}: pH never settled in {no_ph}/{args.num_sessions}, "
            f"EC never settled in {no_ec}/{args.num_sessions}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(
            {
                "n_pairs": args.num_sessions,
                "duration_h": args.duration_h,
                "seed": args.seed,
                "controllers": names,
                "summary": summary,
                "per_session": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved -> {out_path}")


# ---------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hydroponic dosing sessions or compare controllers in closed-loop."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Run sessions and save events to JSON files.")
    g.add_argument("--num_sessions", type=int, default=1)
    g.add_argument("--duration_h", type=float, default=6.0)
    g.add_argument(
        "--controller", choices=AVAILABLE_CONTROLLERS, default="rule",
        help="Which controller to use for dose decisions.",
    )
    g.add_argument("--out_dir", type=str, default="data")
    g.add_argument("--lgbm_model", type=str, default="models/lgbm_inverse.joblib")
    g.add_argument("--nn_model", type=str, default="models/nn_inverse.pt")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("compare", help="Compare controllers on matched-seed pairs.")
    c.add_argument("--num_sessions", type=int, default=50)
    c.add_argument("--duration_h", type=float, default=6.0)
    c.add_argument(
        "--controllers", type=str, default="rule,lgbm",
        help=f"Comma-separated controller names. Available: {','.join(AVAILABLE_CONTROLLERS)}.",
    )
    c.add_argument(
        "--seed", type=int, default=42, help="Master seed for matched-pair generation."
    )
    c.add_argument("--lgbm_model", type=str, default="models/lgbm_inverse.joblib")
    c.add_argument("--nn_model", type=str, default="models/nn_inverse.pt")
    c.add_argument("--out", type=str, default="models/controller_comparison.json")
    c.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
