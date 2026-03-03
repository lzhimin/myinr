"""
Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
Müller et al., SIGGRAPH 2022 — https://arxiv.org/abs/2201.05989

Pure PyTorch implementation of the multiresolution hash encoding + compact MLP.
For production-speed training, replace HashEncoder with the official tinycudann binding:
    https://github.com/NVlabs/tiny-cuda-nn
"""

import torch
import torch.nn as nn


class HashEncoder(nn.Module):
    """
    Multiresolution hash encoding for 3D coordinates.

    Args:
        n_levels:            Number of resolution levels.
        n_features_per_level: Feature dimensions per level.
        log2_hashmap_size:   log2 of the hash table size per level.
        base_resolution:     Coarsest grid resolution.
        finest_resolution:   Finest grid resolution.
    """

    # Spatial hash primes (Müller et al.)
    PI1 = 1
    PI2 = 2_654_435_761
    PI3 = 805_459_861

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size
        self.out_dim = n_levels * n_features_per_level

        # Per-level resolution: geometric progression from base to finest
        growth = (finest_resolution / base_resolution) ** (1.0 / (n_levels - 1))
        self.register_buffer(
            "resolutions",
            torch.tensor([int(base_resolution * (growth ** i)) for i in range(n_levels)], dtype=torch.float32),
        )

        # Learnable hash tables: one embedding per level
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

    def _hash(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (..., 3) integer grid coordinates
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        return ((x * self.PI1) ^ (y * self.PI2) ^ (z * self.PI3)) % self.hashmap_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 3) coordinates normalized to [0, 1].
        Returns:
            (..., n_levels * n_features_per_level) encoded features.
        """
        encoded = []
        for level, (res, emb) in enumerate(zip(self.resolutions, self.embeddings)):
            # Scale coordinates to grid
            scaled = x * res                          # (..., 3)
            floor = scaled.long()                     # lower corner
            ceil = floor + 1                          # upper corner
            frac = scaled - floor.float()             # trilinear weights

            # 8 corners of the voxel
            corners = torch.stack([
                torch.stack([floor[..., 0], floor[..., 1], floor[..., 2]], dim=-1),
                torch.stack([ceil[..., 0],  floor[..., 1], floor[..., 2]], dim=-1),
                torch.stack([floor[..., 0], ceil[..., 1],  floor[..., 2]], dim=-1),
                torch.stack([ceil[..., 0],  ceil[..., 1],  floor[..., 2]], dim=-1),
                torch.stack([floor[..., 0], floor[..., 1], ceil[..., 2]],  dim=-1),
                torch.stack([ceil[..., 0],  floor[..., 1], ceil[..., 2]],  dim=-1),
                torch.stack([floor[..., 0], ceil[..., 1],  ceil[..., 2]],  dim=-1),
                torch.stack([ceil[..., 0],  ceil[..., 1],  ceil[..., 2]],  dim=-1),
            ], dim=0)                                  # (8, ..., 3)

            # Trilinear interpolation weights
            wx = frac[..., 0]; wy = frac[..., 1]; wz = frac[..., 2]
            weights = torch.stack([
                (1 - wx) * (1 - wy) * (1 - wz),
                wx       * (1 - wy) * (1 - wz),
                (1 - wx) * wy       * (1 - wz),
                wx       * wy       * (1 - wz),
                (1 - wx) * (1 - wy) * wz,
                wx       * (1 - wy) * wz,
                (1 - wx) * wy       * wz,
                wx       * wy       * wz,
            ], dim=0)                                  # (8, ...)

            # Hash lookup and interpolate
            hashed = self._hash(corners)               # (8, ...)
            feats = emb(hashed)                        # (8, ..., F)
            interp = (weights.unsqueeze(-1) * feats).sum(dim=0)  # (..., F)
            encoded.append(interp)

        return torch.cat(encoded, dim=-1)              # (..., n_levels * F)


class InstantNGP(nn.Module):
    """
    Instant-NGP: multiresolution hash encoding + compact MLP.

    Args:
        n_levels:            Hash encoding resolution levels.
        n_features_per_level: Features per level.
        log2_hashmap_size:   log2 of hash table size.
        base_resolution:     Coarsest grid resolution.
        finest_resolution:   Finest grid resolution.
        hidden_features:     MLP hidden width.
        hidden_layers:       Number of MLP hidden layers.
        out_features:        Output dimensionality (e.g. 1 for scalar field, 4 for density+color).
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        out_features: int = 4,
    ):
        super().__init__()
        self.encoder = HashEncoder(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )

        in_dim = self.encoder.out_dim
        layers = []
        for _ in range(hidden_layers):
            layers += [nn.Linear(in_dim, hidden_features), nn.ReLU()]
            in_dim = hidden_features
        layers.append(nn.Linear(in_dim, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 3) coordinates normalized to [0, 1].
        Returns:
            (..., out_features) predicted values.
        """
        return self.mlp(self.encoder(x))
