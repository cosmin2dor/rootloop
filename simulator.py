"""Hydroponic tank simulator — physics-based plant model for the dosing project.

State variables:
  V    : tank volume (L)
  m_N  : dissolved nutrient mass (g)
  A    : net alkalinity reservoir (abstract units)

True equilibrium:
  EC_eq = k_EC * (m_N / V)
  pH_eq = pH_min + (pH_max - pH_min) / (1 + exp(-k_pH * A/V))

Sensor-side values lag toward equilibrium with first-order dynamics
(time constants tau_mix_EC, tau_mix_pH). Read() adds Gaussian noise.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random


@dataclass
class SimConfig:
    # Initial state
    V0: float = 5.0           # tank volume (L)
    initial_ec: float = 1.5   # mS/cm
    initial_ph: float = 6.5

    # Chemistry
    C_stock_g_per_L: float = 50.0   # nutrient stock concentration
    k_EC: float = 20.0              # mS/cm per (g/L) — calibrated so 1 ml stock per L tank ~ 1 mS/cm
    pH_min: float = 4.0
    pH_max: float = 9.5
    k_pH: float = 6.0               # buffer steepness (operates on A/V)

    # Dose effects
    beta_up: float = 0.6      # alkalinity per ml of pH-up
    beta_down: float = 0.6    # alkalinity per ml of pH-down
    gamma_NpH: float = 0.05   # alkalinity drop per ml of nutrient (mildly acidic stock)

    # Sensor-side dynamics
    tau_mix_EC_s: float = 20.0
    tau_mix_pH_s: float = 30.0

    # Slow drift (per-hour rates)
    r_uptake_g_per_L_per_h: float = 0.01   # plant nutrient consumption
    r_drift_pH_per_h: float = 0.05         # natural alkaline drift

    # Sensor noise (std dev)
    sigma_EC: float = 0.03
    sigma_pH: float = 0.03


class HydroponicSim:
    def __init__(self, config: SimConfig | None = None, seed: int = 0):
        self.cfg = config or SimConfig()
        self.rng = random.Random(seed)
        c = self.cfg

        self.V = c.V0
        self.m_N = c.initial_ec * self.V / c.k_EC
        ratio = (c.pH_max - c.pH_min) / (c.initial_ph - c.pH_min) - 1.0
        if ratio <= 0:
            raise ValueError(
                f"initial_ph must be strictly between pH_min={c.pH_min} and pH_max={c.pH_max}"
            )
        self.A = -math.log(ratio) * self.V / c.k_pH

        self.ec_meas = self._ec_eq()
        self.ph_meas = self._ph_eq()

    def _ec_eq(self) -> float:
        return self.cfg.k_EC * self.m_N / self.V

    def _ph_eq(self) -> float:
        c = self.cfg
        return c.pH_min + (c.pH_max - c.pH_min) / (1.0 + math.exp(-c.k_pH * self.A / self.V))

    @property
    def ec_eq(self) -> float:
        return self._ec_eq()

    @property
    def ph_eq(self) -> float:
        return self._ph_eq()

    def step(self, dt_s: float) -> None:
        c = self.cfg
        dt_h = dt_s / 3600.0

        self.m_N = max(0.0, self.m_N - c.r_uptake_g_per_L_per_h * self.V * dt_h)
        self.A += c.r_drift_pH_per_h * dt_h

        ec_eq = self._ec_eq()
        ph_eq = self._ph_eq()
        self.ec_meas = ec_eq + (self.ec_meas - ec_eq) * math.exp(-dt_s / c.tau_mix_EC_s)
        self.ph_meas = ph_eq + (self.ph_meas - ph_eq) * math.exp(-dt_s / c.tau_mix_pH_s)

    def dose(
        self,
        nutrient_ml: float = 0.0,
        ph_up_ml: float = 0.0,
        ph_down_ml: float = 0.0,
    ) -> None:
        c = self.cfg
        self.m_N += c.C_stock_g_per_L * nutrient_ml / 1000.0
        self.A -= c.gamma_NpH * nutrient_ml
        self.A += c.beta_up * ph_up_ml
        self.A -= c.beta_down * ph_down_ml
        self.V += (nutrient_ml + ph_up_ml + ph_down_ml) / 1000.0

    def read(self) -> tuple[float, float]:
        c = self.cfg
        ec = self.ec_meas + self.rng.gauss(0.0, c.sigma_EC)
        ph = self.ph_meas + self.rng.gauss(0.0, c.sigma_pH)
        return ec, ph
