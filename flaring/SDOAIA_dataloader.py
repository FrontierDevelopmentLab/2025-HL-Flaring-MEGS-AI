
import torch
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom

class AIA_GOESDataset(torch.utils.data.Dataset):
    """Dataset for loading AIA images and SXR values for regression.
    Uses one-to-one data mapping (no time-series)."""

    def __init__(self, aia_dir, sxr_dir, transform=None, sxr_transform=None, target_size=(224, 224)):
        """
        Initialize dataset with AIA and SXR directories.

        Args:
            aia_dir (str): Path to directory with AIA .npy files.
            sxr_dir (str): Path to directory with SXR .npy files.
            transform (callable, optional): Transform for AIA images.
            sxr_transform (callable, optional): Transform for SXR values.
            target_size (tuple): Target size for AIA images (height, width).
        """
        self.aia_dir = Path(aia_dir)
        self.sxr_dir = Path(sxr_dir)
        self.transform = transform
        self.sxr_transform = sxr_transform
        self.target_size = target_size

        # Collect samples where both AIA and SXR files exist
        self.samples = []
        for f in sorted(self.aia_dir.glob("*_aia.npy")):
            timestamp = f.stem.replace('_aia', '')
            sxr_path = self.sxr_dir / f"{timestamp}_sxr.npy"
            if sxr_path.exists():
                self.samples.append(timestamp)

    def __len__(self):
        """Return the number of valid samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load AIA image and SXR value for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: ((aia_img, sxr_val), target_sxr) where aia_img is the AIA image tensor,
                   sxr_val is the SXR value tensor, and target_sxr is the target SXR tensor.
        """
        timestamp = self.samples[idx]
        aia_path = self.aia_dir / f"{timestamp}_aia.npy"
        sxr_path = self.sxr_dir / f"{timestamp}_sxr.npy"

        # Load AIA (H, W, 6) and transpose to (6, H, W)
        aia_img = np.load(aia_path)  # Shape: (H, W, 6)
        if aia_img.shape[-1] != 6:
            raise ValueError(f"AIA image {aia_path} has unexpected shape {aia_img.shape}")
        aia_img = aia_img.transpose(2, 0, 1)  # Shape: (6, H, W)

        # Resize AIA image
        if aia_img.shape[1:3] != self.target_size:
            aia_img = zoom(aia_img, (1, self.target_size[0]/aia_img.shape[1], self.target_size[1]/aia_img.shape[2]))

        # Load SXR (ensure scalar)
        sxr_val = np.load(sxr_path)
        if sxr_val.size != 1:
            raise ValueError(f"SXR value {sxr_path} has unexpected size {sxr_val.size}, expected scalar")
        sxr_val = np.atleast_1d(sxr_val).flatten()[0]  # Ensure scalar

        # Apply transforms
        if self.transform:
            aia_img = self.transform(aia_img)
            if not isinstance(aia_img, np.ndarray):
                raise TypeError(f"AIA transform must return a NumPy array, got {type(aia_img)}")
        if self.sxr_transform:
            sxr_val = self.sxr_transform(sxr_val)
            if not isinstance(sxr_val, (np.floating, float, np.integer, int)):
                raise TypeError(f"SXR transform must return a scalar number, got {type(sxr_val)}")

        # For regression, target is the same SXR value
        return ((torch.tensor(aia_img, dtype=torch.float32),
                 torch.tensor(sxr_val, dtype=torch.float32)),
                torch.tensor(sxr_val, dtype=torch.float32))