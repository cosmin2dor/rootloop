"""NN architecture for the inverse dosing model.

Imported by both train_nn.py and run_session.py so the architecture lives in
exactly one place.
"""

from __future__ import annotations
import torch
from torch import nn


class DoseNet(nn.Module):
    """MLP: (ec, ph) -> (nutrient_ml, ph_up_ml, ph_down_ml).

    Output is linear (not ReLU); negative predictions are handled by post-hoc
    clipping at inference time, identical to the LightGBM treatment.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
