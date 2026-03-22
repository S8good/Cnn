from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainSpectraDataset(Dataset):
    def __init__(self, spectra_path: str, labels_path: str):
        self.spectra = np.load(spectra_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if self.spectra.shape[0] != self.labels.shape[0]:
            raise ValueError("spectra and labels must have same length")

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def __getitem__(self, idx: int):
        x = self.spectra[idx].astype(np.float32)
        y = int(self.labels[idx])
        x = torch.from_numpy(x).unsqueeze(0)
        return x, y


def split_indices(n: int, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def make_subset(dataset: Dataset, indices: np.ndarray) -> torch.utils.data.Subset:
    return torch.utils.data.Subset(dataset, indices.tolist())
