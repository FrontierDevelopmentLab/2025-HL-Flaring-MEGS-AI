
import numpy as np
from pathlib import Path
import glob
import os

def compute_sxr_norm(sxr_dir):
    """
    Compute mean and standard deviation of log10-transformed SXR values.

    Args:
        sxr_dir (str): Path to directory containing SXR .npy files.

    Returns:
        tuple: (mean, std) of log10(SXR + 1e-8) values.
    """
    sxr_dir = Path(sxr_dir).resolve()
    print(f"Checking SXR directory: {sxr_dir}")
    if not sxr_dir.is_dir():
        raise FileNotFoundError(f"SXR directory does not exist or is not a directory: {sxr_dir}")

    # Use glob for case-insensitive matching
    sxr_files = sorted(glob.glob(os.path.join(sxr_dir, "*.npy")))
    print(f"Found {len(sxr_files)} SXR files in {sxr_dir}")
    if len(sxr_files) == 0:
        print(f"No files matching '*_sxr.npy' found. Listing directory contents:")
        print(os.listdir(sxr_dir)[:10])  # Show first 10 files
        raise ValueError(f"No SXR files found in {sxr_dir}")

    sxr_values = []
    for f in sxr_files:
        try:
            sxr = np.load(f)
            sxr = np.atleast_1d(sxr).flatten()[0]
            if not np.isfinite(sxr) or sxr < 0:
                print(f"Skipping invalid SXR value in {f}: {sxr}")
                continue
            sxr_values.append(np.log10(sxr + 1e-8))
        except Exception as e:
            print(f"Failed to load SXR file {f}: {e}")
            continue

    sxr_values = np.array(sxr_values)
    if len(sxr_values) == 0:
        raise ValueError(f"No valid SXR values found in {sxr_dir}. All files failed to load or contained invalid data.")

    mean = np.mean(sxr_values)
    std = np.std(sxr_values)
    print(f"Computed SXR normalization: mean={mean}, std={std}")
    return mean, std

if __name__ == "__main__":
    # Update this path to your real data SXR directory
    sxr_dir = "/mnt/data3/ML-Ready_clean/non_flares_event_dir/SXR/train/"  # Replace with actual path
    sxr_norm = compute_sxr_norm(sxr_dir)
    np.save("/mnt/data3/ML-Ready_clean/non_flares_event_dir/SXR/normalized_sxr.npy", sxr_norm)
    #print(f"Saved SXR normalization to /mnt/data/ML-Ready-Data-No-Intensity-Cut/normalized_sxr")