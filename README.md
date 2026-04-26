# 🌱 Rootloop

ML-based dosing for hydroponic systems. Given current EC (electrical conductivity)                                                         and pH readings, predict dose volumes for nutrient (A+B mixed), pH-up, and                                                                   pH-down to keep the tank in optimal range.

## Status

Phase 1 (sim pipeline) ✅. Rule, LightGBM, and a small MLP give equivalent closed-loop control on the simulator. Phase 2 (lab) is next.

## Layout

```
simulator.py                  5 L tank physics
run_session.py                session driver + controller comparison
data_loader.py                events → (state, action) pairs
train_lgbm.py / train_nn.py   training
nn_model.py                   DoseNet MLP architecture
schemas/events.schema.json    event format (sim and lab)
data/training, data/test      simulated sessions
models/                       checkpoints + metrics
```

## Quickstart

```bash
uv venv
uv add numpy lightgbm torch joblib

# Sim data
uv run run_session.py generate --controller rule --num_sessions 200 --out_dir data/training
uv run run_session.py generate --controller rule --num_sessions  50 --out_dir data/test

# Train
uv run train_lgbm.py
uv run train_nn.py

# Closed-loop comparison
uv run run_session.py compare --controllers rule,lgbm,nn --num_sessions 50
```

## 🔁 Control loop

Dose → wait 2 min → measure → repeat. The wait absorbs mixing and sensor lag,
so the model can stay stateless: one inference call per cycle, input is just
`(ec, ph)`, output is `(nutrient_ml, ph_up_ml, ph_down_ml)`.

## 🧪 Simulator

5 L recirculating tank. Throwaway — exists only to validate the pipeline before
the lab build.

**State**

- `V` — tank volume (L)
- `m_N` — dissolved nutrient mass (g)
- `A` — net alkalinity reservoir (abstract units)

**Equilibrium**

```
EC_eq = k_EC * m_N / V                                       # linear
pH_eq = pH_min + (pH_max - pH_min) / (1 + exp(-k_pH * A/V))  # buffered sigmoid
```

**Sensors** track equilibrium with first-order lag (τ_EC ≈ 20 s, τ_pH ≈ 30 s) and additive Gaussian noise (σ ≈ 0.03).

**Drift (per hour)**

- `m_N` ↓ — plant uptake
- `A` ↑ — natural alkaline drift

**Dose effects**

- Nutrient adds `m_N`; also drops `A` slightly (stock is mildly acidic → this is the EC↔pH cross-coupling).
- pH-up adds `A`. pH-down subtracts.
- All doses change `V` by the dose volume.

**Out of scope**: temperature, biofilm, sensor drift on day/week scales, pump
quantization, growth-stage variation in plant uptake, ammonium-vs-nitrate
balance. Defaults are tuned for "qualitatively realistic" — absolute numbers
will diverge from any real tank.

## 🎚️ Operating range

| Threshold | pH            | EC (mS/cm)    |
| --------- | ------------- | ------------- |
| Setpoint  | 6.0           | 1.8           |
| Deadband  | 5.7 – 6.3     | 1.6 – 2.0     |
| Safe zone | 5.0 – 7.0     | 1.0 – 3.0     |
| Alarm     | < 4.5 / > 7.5 | < 0.5 / > 4.0 |

## 📊 Phase 1 results

50 matched-pair sessions. Time-to-deadband, % time outside safe zone, steady-state std, total dose volume — all three controllers within paired noise of each other. Both ML models inherit the rule's 2% pH-alarm rate. Test-set R² 0.91–0.98 on dose volumes.

The models learned the rule. To beat the rule needs Phase 2 — analytical-optimal labels or RL.

## 🧬 Phase 2 — lab

Schema is identical for sim and real by design. To swap hardware in:

- Write `LabTank` with the same interface as `HydroponicSim` (`.V`, `.read()`, `.dose()`, `.step(dt)`). `run_session.py` works as-is.
- Add `calibrate_sensors.py`, `calibrate_pumps.py`.
- Discard sim weights, retrain on lab data.

For leafy greens, `ph_up_ml` is rarely useful in practice — plant nitrate uptake drives pH upward, so only pH-down is needed. Drop the channel before training on lab data.
