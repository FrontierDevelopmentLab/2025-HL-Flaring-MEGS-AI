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
        print(f"Initialized dataset with AIA dir: {self.aia_dir}, SXR dir: {self.sxr_dir}")

        # Find matching files
        aia_files = sorted(glob.glob(str(self.aia_dir / "*.npy")))
        aia_files = [Path(f) for f in aia_files]
        print(f"Found {len(aia_files)} AIA files")

        for f in aia_files:
            timestamp = f.stem
            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            if sxr_path.exists():
                self.samples.append(timestamp)

        if len(self.samples) == 0:
            raise ValueError("No valid AIA/SXR sample pairs found")
        print(f"Found {len(self.samples)} valid AIA/SXR pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, timestamp: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single sample given a timestamp."""
        aia_path = self.aia_dir / f"{timestamp}.npy"
        sxr_path = self.sxr_dir / f"{timestamp}.npy"

        # Load AIA image as (6, H, W)
        aia_img = np.load(aia_path)
        if aia_img.shape[0] != 6:
            raise ValueError(f"AIA image {timestamp} has {aia_img.shape[0]} channels, expected 6")
        if aia_img.shape[1:] != self.target_size:
            raise ValueError(f"AIA image {timestamp} shape {aia_img.shape[1:]} does not match target size {self.target_size}")
        aia_img = torch.tensor(aia_img, dtype=torch.float32)
        if self.transform:
            aia_img = self.transform(aia_img)
        # Keep channel-first: [C, H, W]
        #print(f"Loaded AIA image {timestamp}: shape={aia_img.shape}, min={aia_img.min():.2f}, max={aia_img.max():.2f}")

        # Load SXR value
        sxr_val = np.load(sxr_path)
        if sxr_val.size != 1:
            raise ValueError(f"SXR value {timestamp} has size {sxr_val.size}, expected scalar")
        sxr_val = float(np.atleast_1d(sxr_val).flatten()[0])
        if self.sxr_transform:
            sxr_val = self.sxr_transform(sxr_val)
        #print(f"Loaded SXR value {timestamp}: value={sxr_val:.4f}")
        return aia_img, torch.tensor(sxr_val, dtype=torch.float32)

    def __getitem__(self, idx: int):
        timestamp = self.samples[idx]
        return self._load_sample(timestamp)

class AIA_GOESSequenceDataset(AIA_GOESDataset):
    """Dataset for sequence-to-sequence regression with AIA images."""
    def __init__(self, aia_dir: str, sxr_dir: str, sequence_length: int = 12, stride: int = 1,
                 transform=None, sxr_transform=None, target_size: Tuple[int, int] = (512, 512)):
        super().__init__(aia_dir, sxr_dir, transform, sxr_transform, target_size)
        self.sequence_length = sequence_length
        self.stride = stride
        self.sequences = self._create_sequences()
        print(f"Created {len(self.sequences)} sequences with length={sequence_length}, stride={stride}")

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
        """Returns a sequence of AIA images and corresponding SXR values."""
        sequence_timestamps = self.sequences[idx]
        aia_sequence = []
        sxr_values = []

        for timestamp in sequence_timestamps:
            aia_img, sxr_val = self._load_sample(timestamp)
            aia_sequence.append(aia_img)
            sxr_values.append(sxr_val)

        aia_sequence = torch.stack(aia_sequence)  # [T, C, H, W]
        sxr_values = torch.stack(sxr_values)  # [T]
        #print(f"Sequence {idx}: AIA shape={aia_sequence.shape}, SXR shape={sxr_values.shape}")
        return aia_sequence, sxr_values

class AIA_GOESSequenceDataModule(LightningDataModule):
    """DataModule for sequence-to-sequence regression."""
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
        self.train_transforms = train_transforms or T.Compose([])
        self.val_transforms = val_transforms or T.Compose([])
        #(f"DataModule initialized: batch_size={batch_size}, num_workers={num_workers}")

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
        print(f"Dataset sizes: Train={len(self.train_ds)}, Val={len(self.val_ds)}, Test={len(self.test_ds)}")

    def train_dataloader(self):
        loader = DataLoader(self.train_ds, batch_size=self.batch_size,
                            shuffle=True, num_workers=self.num_workers)
       # print(f"Train DataLoader: {len(loader)} batches")
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_ds, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)
        #print(f"Val DataLoader: {len(loader)} batches")
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_ds, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.num_workers)
        #print(f"Test DataLoader: {len(loader)} batches")
        return loader