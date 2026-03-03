"""
SIREN: Implicit Neural Representations with Periodic Activation Functions
Sitzmann et al., NeurIPS 2020 — https://arxiv.org/abs/2006.09661
"""

import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features)

    def _init_weights(self, in_features: int):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN network for implicit neural representation of 3D scientific data.

    Args:
        in_features:   Input dimensionality (e.g. 3 for xyz coordinates).
        out_features:  Output dimensionality (e.g. 1 for scalar field).
        hidden_features: Width of each hidden layer.
        hidden_layers:   Number of hidden sine layers.
        omega_0:         Frequency factor for sine activations.
    """

    def __init__(
        self,
        in_features: int = 3,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        omega_0: float = 30.0,
    ):
        super().__init__()

        layers = [SineLayer(in_features, hidden_features, omega_0=omega_0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
        self.net = nn.Sequential(*layers)

        # Final linear layer (no sine activation)
        self.final = nn.Linear(hidden_features, out_features)
        nn.init.uniform_(self.final.weight, -math.sqrt(6.0 / hidden_features) / omega_0, math.sqrt(6.0 / hidden_features) / omega_0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.net(x))
