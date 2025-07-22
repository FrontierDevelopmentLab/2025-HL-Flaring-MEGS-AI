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

    def __init__(self, aia_dir, sxr_dir, transform=None, sxr_transform=None, target_size=(512, 512)):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        timestamp = self.samples[idx]
        aia_path = self.aia_dir / f"{timestamp}.npy"
        sxr_path = self.sxr_dir / f"{timestamp}.npy"

        # Load AIA image as (6, H, W)
        try:
            aia_img = np.load(aia_path)
        except:
            print(f"Error loading AIA image from {aia_path}. Skipping sample.")
            return None

        if aia_img.shape[0] != 6:
            raise ValueError(f"AIA image has {aia_img.shape[0]} channels, expected 6")

        # Resize if needed (operates on (6, H, W))
        # if aia_img.shape[1:3] != self.target_size:
        #     aia_img = zoom(aia_img, (1,
        #                              self.target_size[0]/aia_img.shape[1],
        #                              self.target_size[1]/aia_img.shape[2]))

       # #Apply cut and normalize:
       #  cuts_dict = {
       #      0: np.float32(16.560747),
       #      1: np.float32(75.84181),
       #      2: np.float32(1536.1443),
       #      3: np.float32(2288.1),
       #      4: np.float32(1163.9178),
       #      5: np.float32(401.82352)
       #  }
       #
       #  for channel in range(6):
       #      aia_img[channel] = np.clip(aia_img[channel], 0, cuts_dict[channel])
       #      aia_img[channel] = aia_img[channel] / cuts_dict[channel]  # Normalize each channel to [0, 1]

        # Convert to torch for transforms
        aia_img = torch.tensor(aia_img, dtype=torch.float32) # (6, H, W)

        # Apply transforms (should expect channel-first (C, H, W))
        if self.transform:
            aia_img = self.transform(aia_img)

        # Always output channel-last for model: (H, W, C)
        aia_img = aia_img.permute(1,2,0) # (H, W, 6)

        # Load SXR value
        sxr_val = np.load(sxr_path)
        if sxr_val.size != 1:
            raise ValueError(f"SXR value has size {sxr_val.size}, expected scalar")
        sxr_val = float(np.atleast_1d(sxr_val).flatten()[0])
        if self.sxr_transform:
            sxr_val = self.sxr_transform(sxr_val)

        return (aia_img, torch.tensor(sxr_val, dtype=torch.float32)), torch.tensor(sxr_val, dtype=torch.float32)

class AIA_GOESDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for AIA and SXR data."""

    def __init__(self, aia_train_dir, aia_val_dir,aia_test_dir,sxr_train_dir,sxr_val_dir,sxr_test_dir, sxr_norm, batch_size=64, num_workers=4,
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


    def setup(self, stage=None):

        self.train_ds = AIA_GOESDataset(
            aia_dir=self.aia_train_dir,
            sxr_dir=self.sxr_train_dir,
            transform=self.train_transforms,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512)
        )

        self.val_ds = AIA_GOESDataset(
            aia_dir=self.aia_val_dir,
            sxr_dir=self.sxr_val_dir,
            transform=self.val_transforms,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512)
        )


        self.test_ds = AIA_GOESDataset(
            aia_dir=self.aia_test_dir,
            sxr_dir=self.sxr_test_dir,
            transform=self.val_transforms,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]),
            target_size=(512, 512)
        )


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
