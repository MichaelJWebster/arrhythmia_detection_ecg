"""Data loading for ecg_classify_beats.ipynb.

ECGDataset lives in a module rather than in the notebook so that DataLoader
worker processes can import it. On macOS (and Windows) workers are started with
"spawn", which re-imports everything from scratch and cannot see classes defined
inside a Jupyter kernel; with num_workers > 0 the workers would crash.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def default_num_workers() -> int:
    """One DataLoader worker per CPU core, leaving one core free."""
    return max(0, (os.cpu_count() or 1) - 1)


def read_ecg_csv(path) -> pd.DataFrame:
    """Read one of the Kaggle heartbeat CSVs (mitbih_*.csv, ptbdb_*.csv).

    The files have no header row: 187 float signal columns followed by an
    integer class label. header=None keeps the first beat as data instead of
    turning it into column names.
    """
    n_cols = pd.read_csv(path, header=None, nrows=1).shape[1]
    dtype_map = {i: np.float32 for i in range(n_cols - 1)}
    dtype_map[n_cols - 1] = np.uint8
    return pd.read_csv(path, header=None, dtype=dtype_map)


class BeatAugment:
    """Random augmentation of one beat, for oversampled training data.

    Kachuee et al. augment the rare classes until the classes are balanced, but do not
    say how; these are common choices for this dataset: amplitude scaling, time
    stretching (the beat is resampled, then cropped or zero-padded back to its length,
    as the CSV beats are zero-padded at the end), and a little Gaussian noise.
    Uses NumPy's global RNG, which Lightning seeds per DataLoader worker.
    """

    def __init__(self, scale=(0.8, 1.2), stretch=(0.9, 1.1), noise_std=0.01):
        self.scale = scale
        self.stretch = stretch
        self.noise_std = noise_std

    def __call__(self, signal: np.ndarray) -> torch.Tensor:
        n = len(signal)
        stretched_n = int(round(n * np.random.uniform(*self.stretch)))
        x = np.interp(np.linspace(0, n - 1, stretched_n), np.arange(n), signal)
        x = x[:n] if stretched_n >= n else np.pad(x, (0, n - stretched_n))
        x = x * np.random.uniform(*self.scale) + np.random.normal(0.0, self.noise_std, n)
        return torch.from_numpy(x.astype(np.float32))


def balanced_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    """Draw training beats with every class equally likely (rare beats are repeated,
    so pair this with BeatAugment). An epoch is still len(labels) draws."""
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels),
                                 replacement=True, generator=torch.Generator().manual_seed(seed))


PTB_CLASSES = {0: "Normal", 1: "MI"}


def load_ptbdb(data_dir) -> pd.DataFrame:
    """PTB Diagnostic beats: ptbdb_normal.csv (label 0) and ptbdb_abnormal.csv
    (label 1, myocardial infarction), concatenated. Same layout as MIT-BIH."""
    from pathlib import Path
    data_dir = Path(data_dir).expanduser()
    frames = [read_ecg_csv(data_dir / "ptbdb_normal.csv"),
              read_ecg_csv(data_dir / "ptbdb_abnormal.csv")]
    return pd.concat(frames, ignore_index=True)


class ECGDataset(Dataset):
    VAL_TO_CLASS = {
        0: "N",
        1: "S",
        2: "V",
        3: "F",
        4: "Q",
    }

    def __init__(self, data: pd.DataFrame, transform=None):
        # Use the frame we are given. (This used to re-read mitbih_train.csv
        # regardless, so the "test" loader was really evaluating training data.)
        self.df = data.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        signal = row.iloc[:-1].to_numpy().astype(np.float32)
        label = int(row.iloc[-1])
        if self.transform:
            signal = self.transform(signal)
        else:
            signal = torch.from_numpy(signal)
        return signal, torch.tensor(label, dtype=torch.long)
