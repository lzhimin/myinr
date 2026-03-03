"""
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
Mildenhall et al., ECCV 2020 — https://arxiv.org/abs/2003.08934
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Maps input coordinates to a higher-dimensional space using sinusoidal functions."""

    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)

    def out_dim(self, in_dim: int) -> int:
        d = 2 * in_dim * self.num_freqs
        if self.include_input:
            d += in_dim
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C)
        enc = [x] if self.include_input else []
        for freq in self.freqs:
            enc.append(torch.sin(freq * x))
            enc.append(torch.cos(freq * x))
        return torch.cat(enc, dim=-1)


class NeRF(nn.Module):
    """
    NeRF MLP for volumetric scene representation.

    Given 3D coordinates (and optionally viewing directions), predicts
    volume density (sigma) and RGB color — the core of the NeRF pipeline.

    Args:
        pos_freqs:       Number of positional encoding frequencies for xyz.
        dir_freqs:       Number of positional encoding frequencies for direction.
        hidden_features: Width of hidden layers.
        hidden_layers:   Total number of hidden layers.
        skip_layer:      Layer index at which to concatenate the encoded position.
        use_viewdirs:    Whether to condition color on viewing direction.
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_features: int = 256,
        hidden_layers: int = 8,
        skip_layer: int = 4,
        use_viewdirs: bool = True,
    ):
        super().__init__()
        self.skip_layer = skip_layer
        self.use_viewdirs = use_viewdirs

        self.pos_enc = PositionalEncoding(num_freqs=pos_freqs)
        self.dir_enc = PositionalEncoding(num_freqs=dir_freqs)

        pos_dim = self.pos_enc.out_dim(3)
        dir_dim = self.dir_enc.out_dim(3)

        # Position MLP
        self.pts_layers = nn.ModuleList()
        in_dim = pos_dim
        for i in range(hidden_layers):
            self.pts_layers.append(nn.Linear(in_dim, hidden_features))
            in_dim = hidden_features + pos_dim if i + 1 == skip_layer else hidden_features

        # Density head
        self.sigma_head = nn.Linear(hidden_features, 1)

        # Color head (conditioned on direction if use_viewdirs)
        if use_viewdirs:
            self.feature_layer = nn.Linear(hidden_features, hidden_features)
            self.color_head = nn.Sequential(
                nn.Linear(hidden_features + dir_dim, hidden_features // 2),
                nn.ReLU(),
                nn.Linear(hidden_features // 2, 3),
                nn.Sigmoid(),
            )
        else:
            self.color_head = nn.Sequential(
                nn.Linear(hidden_features, 3),
                nn.Sigmoid(),
            )

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, positions: torch.Tensor, directions: torch.Tensor = None) -> dict:
        """
        Args:
            positions:  (..., 3) 3D coordinates.
            directions: (..., 3) unit viewing directions (required if use_viewdirs=True).

        Returns:
            dict with keys:
                'sigma': (..., 1) volume density
                'rgb':   (..., 3) color
        """
        pos_enc = self.pos_enc(positions)
        x = pos_enc

        for i, layer in enumerate(self.pts_layers):
            x = self.relu(layer(x))
            if i + 1 == self.skip_layer:
                x = torch.cat([x, pos_enc], dim=-1)

        sigma = self.softplus(self.sigma_head(x))

        if self.use_viewdirs and directions is not None:
            x = self.feature_layer(x)
            dir_enc = self.dir_enc(directions)
            x = torch.cat([x, dir_enc], dim=-1)

        rgb = self.color_head(x)

        return {"sigma": sigma, "rgb": rgb}
