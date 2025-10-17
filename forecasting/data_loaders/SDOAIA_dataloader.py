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

    def __init__(self, aia_dir, sxr_dir, wavelengths=[94, 131, 171, 193, 211, 304], sxr_transform=None,
                 target_size=(512, 512), cadence=1, reference_time=None, only_prediction=False, oversample=False,
                 flare_threshold=1e-5, balance_strategy='upsample_minority', sxr_channels=['a', 'b']):
        self.aia_dir = Path(aia_dir).resolve()
        self.sxr_dir = Path(sxr_dir).resolve()
        self.wavelengths = wavelengths
        self.sxr_transform = sxr_transform
        self.target_size = target_size
        self.samples = []
        self.only_prediction = only_prediction
        self.cadence = timedelta(minutes=cadence)
        self.reference_time = reference_time
        self.oversample = oversample
        self.flare_threshold = flare_threshold
        self.balance_strategy = balance_strategy  # 'upsample_minority', 'downsample_majority', 'balanced'
        self.sxr_channels = sxr_channels  # ['a', 'b'] or ['b'] (never ['a'] only)
        self.sxr_norm = None  # Will be set by DataModule
        
        # Validate channel configuration
        if sxr_channels == ['a']:
            raise ValueError("Cannot train on SXR-A only. Must include SXR-B or train on SXR-B only.")
        if 'b' not in sxr_channels:
            raise ValueError("SXR-B channel must be included in sxr_channels.")

        # Check directories
        if not self.aia_dir.is_dir():
            raise FileNotFoundError(f"AIA directory not found: {self.aia_dir}")
        if not self.sxr_dir.is_dir():
            raise FileNotFoundError(f"SXR directory not found: {self.sxr_dir}")

        # Find matching files
        aia_files = sorted(glob.glob(str(self.aia_dir / "*.npy")))
        aia_files = [Path(f) for f in aia_files]

        # First pass: collect valid samples
        valid_samples = []
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

            if valid_samples and (
                    timestamp_dt - pd.to_datetime(valid_samples[-1])).total_seconds() < self.cadence.total_seconds():
                continue

            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            if sxr_path.exists() and not self.only_prediction:
                valid_samples.append(timestamp)
            elif self.only_prediction:
                valid_samples.append(timestamp)

        # Apply oversampling if requested
        if self.oversample and not self.only_prediction:
            self.samples = self._apply_oversampling(valid_samples)
        else:
            self.samples = valid_samples

        if len(self.samples) == 0 and not self.only_prediction:
            raise ValueError("No valid sample pairs found")

    def _apply_oversampling(self, samples):
        """Apply class-balanced oversampling based on actual class counts"""
        import random

        flare_samples = []
        non_flare_samples = []

        # Separate samples by class
        for timestamp in samples:
            sxr_path = self.sxr_dir / f"{timestamp}.npy"
            try:
                sxr_vals = np.load(sxr_path)
                
                # Handle both scalar and array cases for flare classification
                if sxr_vals.size == 1:
                    sxr_val = float(np.atleast_1d(sxr_vals).flatten()[0])
                elif sxr_vals.size == 2:
                    # Use SXR-B (index 1) for flare classification as it's typically more sensitive
                    sxr_val = float(sxr_vals[1])
                else:
                    # Invalid format, treat as non-flare
                    non_flare_samples.append(timestamp)
                    continue

                if sxr_val > self.flare_threshold:
                    flare_samples.append(timestamp)
                else:
                    non_flare_samples.append(timestamp)
            except:
                # If can't load SXR, treat as non-flare
                non_flare_samples.append(timestamp)

        print(f"Original distribution:")
        print(f"  Non-flare samples: {len(non_flare_samples)}")
        print(f"  Flare samples (>{self.flare_threshold}): {len(flare_samples)}")

        # Apply balancing strategy
        if self.balance_strategy == 'upsample_minority':
            # Upsample minority class to match majority
            if len(flare_samples) < len(non_flare_samples):
                # Flares are minority - oversample them
                target_count = len(non_flare_samples)
                balanced_samples = non_flare_samples.copy()
                balanced_samples.extend(self._oversample_to_count(flare_samples, target_count))
            else:
                # Non-flares are minority - oversample them
                target_count = len(flare_samples)
                balanced_samples = flare_samples.copy()
                balanced_samples.extend(self._oversample_to_count(non_flare_samples, target_count))

        elif self.balance_strategy == 'downsample_majority':
            # Downsample majority class to match minority
            if len(flare_samples) < len(non_flare_samples):
                # Downsample non-flares to match flares
                target_count = len(flare_samples)
                balanced_samples = flare_samples.copy()
                balanced_samples.extend(random.sample(non_flare_samples, target_count))
            else:
                # Downsample flares to match non-flares
                target_count = len(non_flare_samples)
                balanced_samples = non_flare_samples.copy()
                balanced_samples.extend(random.sample(flare_samples, target_count))

        elif self.balance_strategy == 'balanced':
            # Balance both classes to the average count
            total_samples = len(flare_samples) + len(non_flare_samples)
            target_count = total_samples // 2

            balanced_samples = []
            balanced_samples.extend(self._oversample_to_count(flare_samples, target_count))
            balanced_samples.extend(self._oversample_to_count(non_flare_samples, target_count))

        else:
            raise ValueError(f"Unknown balance_strategy: {self.balance_strategy}")

        # Shuffle to mix classes
        random.shuffle(balanced_samples)

        # Count final distribution
        final_flare_count = sum(1 for ts in balanced_samples if self._is_flare_sample(ts))
        final_non_flare_count = len(balanced_samples) - final_flare_count

        print(f"Balanced distribution:")
        print(f"  Non-flare samples: {final_non_flare_count}")
        print(f"  Flare samples: {final_flare_count}")
        print(f"  Total samples: {len(balanced_samples)}")
        print(f"  Balance ratio: {final_flare_count / final_non_flare_count:.2f}")

        return balanced_samples

    def _oversample_to_count(self, samples, target_count):
        """Oversample a list of samples to reach target_count"""
        if len(samples) == 0:
            return []

        import random
        result = samples.copy()

        while len(result) < target_count:
            # Add samples randomly with replacement
            remaining_needed = target_count - len(result)
            to_add = min(remaining_needed, len(samples))
            result.extend(random.choices(samples, k=to_add))

        return result[:target_count]  # Ensure exact count

    def _is_flare_sample(self, timestamp):
        """Check if a timestamp corresponds to a flare sample"""
        sxr_path = self.sxr_dir / f"{timestamp}.npy"
        try:
            sxr_vals = np.load(sxr_path)
            
            # Handle both scalar and array cases for flare classification
            if sxr_vals.size == 1:
                sxr_val = float(np.atleast_1d(sxr_vals).flatten()[0])
            elif sxr_vals.size == 2:
                # Use SXR-B (index 1) for flare classification as it's typically more sensitive
                sxr_val = float(sxr_vals[1])
            else:
                return False
                
            return sxr_val > self.flare_threshold
        except:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        timestamp = self.samples[idx]
        aia_path = self.aia_dir / f"{timestamp}.npy"
        sxr_path = self.sxr_dir / f"{timestamp}.npy"

        # Load AIA image as (6, H, W)
        try:
            all_wavelengths = [94, 131, 171, 193, 211, 304]
            aia_img = np.load(aia_path)
            # Extract dimensions from aia data according to selected wavelengths
            indices = [all_wavelengths.index(wav) for wav in self.wavelengths if wav in all_wavelengths]
            aia_img = aia_img[indices]
        except:
            print(f"Error loading AIA image from {aia_path}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self))

        # Convert to torch for transforms
        aia_img = torch.tensor(aia_img, dtype=torch.float32)  # (6, H, W)
        # Always output channel-last for model: (H, W, C)
        aia_img = aia_img.permute(1, 2, 0)  # (H, W, 6)

        # Load SXR values (A/B)
        if not self.only_prediction:
            sxr_vals = np.load(sxr_path)
        else:
            sxr_vals = np.array([0, 0])  # Default values for prediction mode
        
        # Handle both scalar and array cases
        if sxr_vals.size == 1:
            # Single value case - duplicate for A/B
            sxr_val = float(np.atleast_1d(sxr_vals).flatten()[0])
            sxr_a, sxr_b = sxr_val, sxr_val
        elif sxr_vals.size == 2:
            # Array case - [SXR-A, SXR-B]
            sxr_a = float(sxr_vals[0])
            sxr_b = float(sxr_vals[1])
        else:
            raise ValueError(f"SXR value has size {sxr_vals.size}, expected scalar or [SXR-A, SXR-B]")
        
        # Apply normalization based on configuration
        if self.sxr_transform:
            # Check if we have separate normalization for each channel
            if hasattr(self, 'sxr_norm') and isinstance(self.sxr_norm, dict):
                # Separate normalization for each channel
                sxr_a = (np.log10(sxr_a + 1e-8) - self.sxr_norm['a'][0]) / self.sxr_norm['a'][1]
                sxr_b = (np.log10(sxr_b + 1e-8) - self.sxr_norm['b'][0]) / self.sxr_norm['b'][1]
            else:
                # Single normalization (backward compatibility)
                sxr_a = self.sxr_transform(sxr_a)
                sxr_b = self.sxr_transform(sxr_b)

        # Return only the requested channels
        selected_sxr = []
        if 'a' in self.sxr_channels:
            selected_sxr.append(sxr_a)
        if 'b' in self.sxr_channels:
            selected_sxr.append(sxr_b)
        
        if not selected_sxr:
            raise ValueError(f"No valid SXR channels selected. Available: {self.sxr_channels}")

        return aia_img, torch.tensor(selected_sxr, dtype=torch.float32)
    def __gettimestamp__(self, idx):
        timestamp = self.samples[idx]
        return timestamp

class AIA_GOESDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for AIA and SXR data."""

    def __init__(self, aia_train_dir, aia_val_dir,aia_test_dir,sxr_train_dir,sxr_val_dir,sxr_test_dir, sxr_norm, batch_size=64, num_workers=4, wavelengths=[94,131,171,193, 211,304], cadence = 1, reference_time = None, only_prediction=False, oversample=False, balance_strategy='upsample_minority', sxr_channels=['a', 'b']):
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
        self.cadence = cadence
        self.reference_time = reference_time
        self.only_prediction = only_prediction
        self.oversample = oversample
        self.balance_strategy = balance_strategy
        self.sxr_channels = sxr_channels


    def setup(self, stage=None):

        self.train_ds = AIA_GOESDataset(
            aia_dir=self.aia_train_dir,
            sxr_dir=self.sxr_train_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]) if not isinstance(self.sxr_norm, dict) else None,
            target_size=(512, 512),
            wavelengths= self.wavelengths,
            cadence = 1,
            reference_time = None,
            only_prediction = False,
            oversample = self.oversample,
            flare_threshold = 1e-5,
            balance_strategy = self.balance_strategy,
            sxr_channels = self.sxr_channels
        )
        # Set the normalization parameters for the dataset
        self.train_ds.sxr_norm = self.sxr_norm

        self.val_ds = AIA_GOESDataset(
            aia_dir=self.aia_val_dir,
            sxr_dir=self.sxr_val_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]) if not isinstance(self.sxr_norm, dict) else None,
            target_size=(512, 512),
            wavelengths=self.wavelengths,
            cadence = 1,
            reference_time = None,
            only_prediction = False,
            oversample = False,
            flare_threshold = 1e-5,
            balance_strategy = 'upsample_minority',
            sxr_channels = self.sxr_channels
        )
        # Set the normalization parameters for the dataset
        self.val_ds.sxr_norm = self.sxr_norm


        self.test_ds = AIA_GOESDataset(
            aia_dir=self.aia_test_dir,
            sxr_dir=self.sxr_test_dir,
            sxr_transform=T.Lambda(lambda x: (np.log10(x + 1e-8) - self.sxr_norm[0]) / self.sxr_norm[1]) if not isinstance(self.sxr_norm, dict) else None,
            target_size=(512, 512),
            wavelengths=self.wavelengths,
            cadence = 1,
            reference_time = None,
            only_prediction = False,
            oversample = False,
            flare_threshold = 1e-5,
            balance_strategy = 'upsample_minority',
            sxr_channels = self.sxr_channels
        )
        # Set the normalization parameters for the dataset
        self.test_ds.sxr_norm = self.sxr_norm


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, prefetch_factor= 4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, prefetch_factor= 4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, prefetch_factor= 1)
