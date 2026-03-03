import torch
from torch.utils.data import Dataset


class ScalarFieldDataset(Dataset):
    """
    Dataset of 3D coordinate-value pairs for scalar field regression.

    Args:
        coords: (N, 3) tensor of 3D coordinates, normalized to [0, 1].
        values: (N, D) tensor of target values at each coordinate.
    """

    def __init__(self, coords: torch.Tensor, values: torch.Tensor):
        assert coords.shape[0] == values.shape[0], "coords and values must have the same length"
        self.coords = coords
        self.values = values

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx: int):
        return self.coords[idx], self.values[idx]
