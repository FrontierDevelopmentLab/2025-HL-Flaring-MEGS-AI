import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
import glob
import os
from typing import Optional, Tuple, List

class AIA_GOESDataset(torch.utils.data.Dataset):
    """Base dataset for loading AIA images and SXR values."""

    def __init__(self, aia_dir: str, sxr_dir: str, transform=None, sxr_transform=None,
                 target_size: Tuple[int, int] = (512, 512)):
        self.aia_dir = Path(aia_dir).resolve()
        self.sxr_dir = Path(sxr_dir).resolve()
        self.transform = transform
        self.sxr_transform = sxr_transform
        self.target_size = target_size
        self.samples = []

        # Check directories
        if not self.aia_dir.is_dir():
            raise FileNotFoundError(f"AIA directory not found: {self.aia_dir}")
        if not self.sxr_dir.is_dir():
            raise FileNotFoundError(f"SXR directory not found: {self.sxr_dir}")

        # Find matching files
        aia_files = sorted(glob.glob(str(self.aia_dir / "*.npy")))
        aia_files = [Path(f) for f in aia_files]

        for f in aia_files:
            timestamp = f.stem
            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            if sxr_path.exists():
                self.samples.append(timestamp)

        if len(self.samples) == 0:
            raise ValueError("No valid sample pairs found")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, timestamp: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single sample given a timestamp."""
        aia_path = self.aia_dir / f"{timestamp}.npy"
        sxr_path = self.sxr_dir / f"{timestamp}.npy"

        # Load AIA image as (6, H, W)
        aia_img = np.load(aia_path)
        if aia_img.shape[0] != 6:
            raise ValueError(f"AIA image has {aia_img.shape[0]} channels, expected 6")

        # Convert to torch and apply transforms
        aia_img = torch.tensor(aia_img, dtype=torch.float32)
        if self.transform:
            aia_img = self.transform(aia_img)
        aia_img = aia_img.permute(1, 2, 0)  # (H, W, 6)

        # Load SXR value
        sxr_val = np.load(sxr_path)
        if sxr_val.size != 1:
            raise ValueError(f"SXR value has size {sxr_val.size}, expected scalar")
        sxr_val = float(np.atleast_1d(sxr_val).flatten()[0])
        if self.sxr_transform:
            sxr_val = self.sxr_transform(sxr_val)

        return aia_img, torch.tensor(sxr_val, dtype=torch.float32)

    def __getitem__(self, idx: int):
        """Base implementation for single sample loading."""
        timestamp = self.samples[idx]
        return self._load_sample(timestamp)

class AIA_GOESSequenceDataset(AIA_GOESDataset):
    """Dataset for time-series regression with sequence of AIA images."""

    def __init__(self, aia_dir: str, sxr_dir: str, sequence_length: int = 12, stride: int = 1,
                 transform=None, sxr_transform=None, target_size: Tuple[int, int] = (512, 512)):
        super().__init__(aia_dir, sxr_dir, transform, sxr_transform, target_size)
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = self._create_sequences()

    def _create_sequences(self) -> List[List[str]]:
        """Group timestamps into sequences."""
        sequences = []
        for i in range(0, len(self.samples) - self.sequence_length + 1, self.stride):
            sequence = self.samples[i:i + self.sequence_length]
            sequences.append(sequence)
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sequence of AIA images and the corresponding SXR value (for the last frame)."""
        sequence_timestamps = self.sequences[idx]

        # Load all images in the sequence
        aia_sequence = []
        sxr_values = []

        for timestamp in sequence_timestamps:
            aia_img, sxr_val = self._load_sample(timestamp)
            aia_sequence.append(aia_img)
            sxr_values.append(sxr_val)

        # Stack images along new dimension (T, H, W, C)
        aia_sequence = torch.stack(aia_sequence)

        # For sequence-to-one prediction, we use the last SXR value
        # For sequence-to-sequence, we would return all sxr_values
        return aia_sequence, sxr_values[-1]

class AIA_GOESDataModule(LightningDataModule):
    """DataModule for one-to-one regression."""

    def __init__(self, aia_train_dir: str, aia_val_dir: str, aia_test_dir: str,
                 sxr_train_dir: str, sxr_val_dir: str, sxr_test_dir: str,
                 sxr_norm: Tuple[float, float], batch_size: int = 64, num_workers: int = 4,
                 train_transforms=None, val_transforms=None):
        super().__init__()
        self.aia_train_dir = aia_train_dir
        self.aia_val_dir = aia_val_dir
        self.aia_test_dir = aia_test_dir
        self.sxr_train_dir = sxr_train_dir
        self.sxr_val_dir = sxr_val_dir
        self.sxr_test_dir = sxr_test_dir
        self.sxr_norm = sxr_norm
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def setup(self, stage: Optional[str] = None):
        sxr_transform = T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1])

        self.train_ds = AIA_GOESDataset(
            aia_dir=self.aia_train_dir,
            sxr_dir=self.sxr_train_dir,
            transform=self.train_transforms,
            sxr_transform=sxr_transform,
            target_size=(512, 512))

        self.val_ds = AIA_GOESDataset(
            aia_dir=self.aia_val_dir,
            sxr_dir=self.sxr_val_dir,
            transform=self.val_transforms,
            sxr_transform=sxr_transform,
            target_size=(512, 512))

        self.test_ds = AIA_GOESDataset(
            aia_dir=self.aia_test_dir,
            sxr_dir=self.sxr_test_dir,
            transform=self.val_transforms,
            sxr_transform=sxr_transform,
            target_size=(512, 512))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

class AIA_GOESSequenceDataModule(LightningDataModule):
    """DataModule for time-series regression."""

    def __init__(self, aia_train_dir: str, aia_val_dir: str, aia_test_dir: str,
                 sxr_train_dir: str, sxr_val_dir: str, sxr_test_dir: str,
                 sxr_norm: Tuple[float, float], sequence_length: int = 12, stride: int = 1,
                 batch_size: int = 32, num_workers: int = 4,
                 train_transforms=None, val_transforms=None):
        super().__init__()
        self.aia_train_dir = aia_train_dir
        self.aia_val_dir = aia_val_dir
        self.aia_test_dir = aia_test_dir
        self.sxr_train_dir = sxr_train_dir
        self.sxr_val_dir = sxr_val_dir
        self.sxr_test_dir = sxr_test_dir
        self.sxr_norm = sxr_norm
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def setup(self, stage: Optional[str] = None):
        sxr_transform = T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1])

        self.train_ds = AIA_GOESSequenceDataset(
            aia_dir=self.aia_train_dir,
            sxr_dir=self.sxr_train_dir,
            sequence_length=self.sequence_length,
            stride=self.stride,
            transform=self.train_transforms,
            sxr_transform=sxr_transform,
            target_size=(512, 512))

        self.val_ds = AIA_GOESSequenceDataset(
            aia_dir=self.aia_val_dir,
            sxr_dir=self.sxr_val_dir,
            sequence_length=self.sequence_length,
            stride=self.stride,
            transform=self.val_transforms,
            sxr_transform=sxr_transform,
            target_size=(512, 512))

        self.test_ds = AIA_GOESSequenceDataset(
            aia_dir=self.aia_test_dir,
            sxr_dir=self.sxr_test_dir,
            sequence_length=self.sequence_length,
            stride=self.stride,
            transform=self.val_transforms,
            sxr_transform=sxr_transform,
            target_size=(512, 512))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)