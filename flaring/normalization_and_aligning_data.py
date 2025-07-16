import os
import re
from collections import defaultdict
import numpy as np
from astropy.io import fits
import warnings
import pandas as pd
from astropy.visualization import ImageNormalize, AsinhStretch

warnings.filterwarnings('ignore')

import pandas as pd

# Directory paths for each wavelength folder.
wavelength_dirs = {
    "94": "/mnt/data2/AIA_processed_data/94",
    "131": "/mnt/data2/AIA_processed_data/131",
    "171": "/mnt/data2/AIA_processed_data/171",
    "193": "/mnt/data2/AIA_processed_data/193",
    "211": "/mnt/data2/AIA_processed_data/211",
    "304": "/mnt/data2/AIA_processed_data/304"
}

# Regular expression to extract timestamp from file names.
# Adjust this pattern to match your file naming scheme.
timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

# Collect timestamps found in each wavelength directory.
timestamps_found = defaultdict(set)

for wavelength, dir_path in wavelength_dirs.items():
    try:
        for filename in os.listdir(dir_path):
            match = timestamp_pattern.search(filename)
            if match:
                ts = match.group(0)
                timestamps_found[ts].add(wavelength)
    except Exception as e:
        print(f"Could not read directory {dir_path}: {e}")

# Identify timestamps that exist in all wavelength folders.
all_wavelengths = set(wavelength_dirs.keys())
common_timestamps = [ts for ts, waves in timestamps_found.items() if waves == all_wavelengths]

# Identify which timestamps are missing files for some wavelengths.
missing_files = {
    ts: list(all_wavelengths - waves)
    for ts, waves in timestamps_found.items() if waves != all_wavelengths
}

print("Timestamps present in all wavelength folders:")
for ts in sorted(common_timestamps):
    print(ts)

print("\nTimestamps with missing wavelength files:")
for ts, missing in missing_files.items():
    print(f"{ts}: missing {', '.join(sorted(missing))}")


goes = pd.read_csv("/mnt/data/goes_combined/combined_g18_avg1m_20230701_20230815.csv")
# Convert 'time' column to datetime
goes['time'] = pd.to_datetime(goes['time'], format='%Y-%m-%d %H:%M:%S')


# Initialize the array to store all wavelength data
data_shape = (6, 512, 512)


# Map wavelengths to array indices
wavelength_to_idx = {
    '94': 0,
    '131': 1,
    '171': 2,
    '193': 3,
    '211': 4,
    '304': 5
}

sdo_norms = {0: ImageNormalize(vmin=0, vmax= np.float32(16.560747), stretch=AsinhStretch(0.005), clip=True),
             1: ImageNormalize(vmin=0, vmax= np.float32(75.84181), stretch=AsinhStretch(0.005), clip=True),
             2: ImageNormalize(vmin=0, vmax= np.float32(1536.1443), stretch=AsinhStretch(0.005), clip=True),
             3: ImageNormalize(vmin=0, vmax= np.float32(2288.1), stretch=AsinhStretch(0.005), clip=True),
             4: ImageNormalize(vmin=0, vmax=np.float32(1163.9178), stretch=AsinhStretch(0.005), clip=True),
             5: ImageNormalize(vmin=0, vmax=np.float32(401.82352), stretch=AsinhStretch(0.001), clip=True),
             }



# Load data for each timestamp and wavelength
for time_idx, timestamp in enumerate(common_timestamps):
    sxr = goes[goes['time'] == pd.to_datetime(timestamp)]
    sxr_a = sxr['xrsa_flux'].values[0] if not sxr.empty else None
    sxr_b = sxr['xrsb_flux'].values[0] if not sxr.empty else None
    if sxr_a is None or sxr_b is None:
        print(f"Missing SXR data for timestamp {timestamp}, skipping...")
        continue
    wavelength_data = np.zeros(data_shape, dtype=np.float32)
    sxr_a_data = np.zeros(1, dtype=np.float32)
    sxr_b_data = np.zeros(1, dtype=np.float32)
    sxr_a_data[0] = sxr_a if sxr_a is not None else np.nan
    sxr_b_data[0] = sxr_b if sxr_b is not None else np.nan
    print(f"Processing timestamp: {timestamp} (Index: {time_idx})")
    for wavelength, wave_idx in wavelength_to_idx.items():
        filepath = os.path.join(wavelength_dirs[wavelength], f"{timestamp}.fits")
        with fits.open(filepath) as hdul:
            raw_data = hdul[0].data

            # Apply the appropriate normalization for this wavelength
            if wave_idx in sdo_norms:
                # Get the normalizer for this wavelength index
                normalizer = sdo_norms[wave_idx]

                # Apply normalization and convert to [-1, 1] range
                normalized_data = normalizer(raw_data)
                wavelength_data[wave_idx] = normalized_data * 2 - 1
            else:
                # Fallback if no normalizer exists for this wavelength
                print(f"Warning: No normalizer found for wavelength index {wave_idx}")
                wavelength_data[wave_idx] = raw_data

    # Store the wavelength data for this timestamp
    np.save(f"/mnt/data2/ML-Ready/AIA-Data/{timestamp}.npy", wavelength_data)
    # Store the SXR data
    np.save(f"/mnt/data2/ML-Ready/GOES-18-SXR-A/{timestamp}.npy", sxr_a_data)
    np.save(f"/mnt/data2/ML-Ready/GOES-18-SXR-B/{timestamp}.npy", sxr_b_data)
    print(f"Saved data for timestamp {timestamp} to disk.")
    print(f"Percent: {time_idx + 1} / {len(common_timestamps)}")