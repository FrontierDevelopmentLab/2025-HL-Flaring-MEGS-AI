from datetime import timedelta

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
import glob
import os


class AIA_GOESDataset(torch.utils.data.Dataset):
    """Dataset for loading AIA images and SXR values for regression."""

    def __init__(self, aia_dir, sxr_dir, wavelengths = [94,131,171,211,293,304], sxr_transform=None, target_size=(512, 512), cadence = 1, reference_time = None, only_prediction=False):
        self.aia_dir = Path(aia_dir).resolve()
        self.sxr_dir = Path(sxr_dir).resolve()
        self.wavelengths = wavelengths
        self.sxr_transform = sxr_transform
        self.target_size = target_size
        self.samples = []
        self.only_prediction = only_prediction
        self.cadence = timedelta(minutes=cadence)
        self.reference_time = reference_time

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
            timestamp_dt = pd.to_datetime(timestamp)

            if self.reference_time is None:
                # First sample sets the alignment
                self.reference_time = timestamp_dt
                aligned = True
            else:
                delta = (timestamp_dt - self.reference_time).total_seconds()
                aligned = (delta % self.cadence.total_seconds()) == 0

            if not aligned:
                continue

            if self.samples and (
                    timestamp_dt - pd.to_datetime(self.samples[-1])).total_seconds() < self.cadence.total_seconds():
                continue

            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            if sxr_path.exists() and not self.only_prediction:
                self.samples.append(timestamp)
            elif self.only_prediction:
                self.samples.append(timestamp)


        if len(self.samples) == 0 and not self.only_prediction:
            raise ValueError("No valid sample pairs found")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        timestamp = self.samples[idx]
        aia_path = self.aia_dir / f"{timestamp}.npy"
        sxr_path = self.sxr_dir / f"{timestamp}.npy"

        # Load AIA image as (6, H, W)
        try:
            all_wavelengths = [94,131,171,211,293,304]
            aia_img = np.load(aia_path)
            #Extract dimensions from aia data according to selected wavelengths
            indices = [all_wavelengths.index(wav) for wav in self.wavelengths if wav in all_wavelengths]
            aia_img = aia_img[indices]
        except:
            print(f"Error loading AIA image from {aia_path}. Skipping sample.")
            return None

        # Convert to torch for transforms
        aia_img = torch.tensor(aia_img, dtype=torch.float32) # (6, H, W)
        # Always output channel-last for model: (H, W, C)
        aia_img = aia_img.permute(1,2,0) # (H, W, 6)

        # Load SXR value
        if not self.only_prediction:
            sxr_val = np.load(sxr_path)
        else:
            sxr_val = np.array([0])
        if sxr_val.size != 1:
            raise ValueError(f"SXR value has size {sxr_val.size}, expected scalar")
        sxr_val = float(np.atleast_1d(sxr_val).flatten()[0])
        if self.sxr_transform:
            sxr_val = self.sxr_transform(sxr_val)

        return aia_img, torch.tensor(sxr_val, dtype=torch.float32)

class AIA_GOESDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for AIA and SXR data."""

    def __init__(self, aia_train_dir, aia_val_dir,aia_test_dir,sxr_train_dir,sxr_val_dir,sxr_test_dir, sxr_norm, batch_size=64, num_workers=4, wavelengths=[94,131,171,211,293,304]):
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
        self.wavelengths = wavelengths


    def setup(self, stage=None):

        self.train_ds = AIA_GOESDataset(
            aia_dir=self.aia_train_dir,
            sxr_dir=self.sxr_train_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512),
            wavelengths= self.wavelengths
        )

        self.val_ds = AIA_GOESDataset(
            aia_dir=self.aia_val_dir,
            sxr_dir=self.sxr_val_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512),
            wavelengths=self.wavelengths
        )


        self.test_ds = AIA_GOESDataset(
            aia_dir=self.aia_test_dir,
            sxr_dir=self.sxr_test_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512),
            wavelengths=self.wavelengths
        )


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, prefetch_factor= 4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, prefetch_factor= 4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, prefetch_factor= 1)
