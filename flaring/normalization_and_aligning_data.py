import os
import re
from collections import defaultdict
import numpy as np
from astropy.io import fits
import warnings
import pandas as pd
from astropy.visualization import ImageNormalize, AsinhStretch
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

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
timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

# Map wavelengths to array indices
wavelength_to_idx = {
    '94': 0,
    '131': 1,
    '171': 2,
    '193': 3,
    '211': 4,
    '304': 5
}

# Initialize the array to store all wavelength data
data_shape = (6, 512, 512)

sdo_norms = {
    0: ImageNormalize(vmin=0, vmax=np.float32(16.560747), stretch=AsinhStretch(0.005), clip=True),
    1: ImageNormalize(vmin=0, vmax=np.float32(75.84181), stretch=AsinhStretch(0.005), clip=True),
    2: ImageNormalize(vmin=0, vmax=np.float32(1536.1443), stretch=AsinhStretch(0.005), clip=True),
    3: ImageNormalize(vmin=0, vmax=np.float32(2288.1), stretch=AsinhStretch(0.005), clip=True),
    4: ImageNormalize(vmin=0, vmax=np.float32(1163.9178), stretch=AsinhStretch(0.005), clip=True),
    5: ImageNormalize(vmin=0, vmax=np.float32(401.82352), stretch=AsinhStretch(0.001), clip=True),
}


def process_timestamp(args):
    """
    Process a single timestamp: load wavelength data, apply normalization,
    and save to disk along with SXR data.
    """
    timestamp, goes_data = args
    try:
        # Get SXR data for this timestamp
        sxr = goes_data[goes_data['time'] == pd.to_datetime(timestamp)]
        sxr_a = sxr['xrsa_flux'].values[0] if not sxr.empty else None
        sxr_b = sxr['xrsb_flux'].values[0] if not sxr.empty else None

        if sxr_a is None or sxr_b is None:
            return (timestamp, False, f"Missing SXR data for timestamp {timestamp}")

        # Initialize arrays
        wavelength_data = np.zeros(data_shape, dtype=np.float32)
        sxr_a_data = np.zeros(1, dtype=np.float32)
        sxr_b_data = np.zeros(1, dtype=np.float32)
        sxr_a_data[0] = sxr_a
        sxr_b_data[0] = sxr_b

        # Process each wavelength
        for wavelength, wave_idx in wavelength_to_idx.items():
            filepath = os.path.join(wavelength_dirs[wavelength], f"{timestamp}.fits")

            with fits.open(filepath) as hdul:
                raw_data = hdul[0].data

                # Apply the appropriate normalization for this wavelength
                if wave_idx in sdo_norms:
                    normalizer = sdo_norms[wave_idx]
                    normalized_data = normalizer(raw_data)
                    wavelength_data[wave_idx] = normalized_data * 2 - 1
                else:
                    wavelength_data[wave_idx] = raw_data

        # Save data to disk
        np.save(f"/mnt/data2/ML-Ready/AIA-Data/{timestamp}.npy", wavelength_data)
        np.save(f"/mnt/data2/ML-Ready/GOES-18-SXR-A/{timestamp}.npy", sxr_a_data)
        np.save(f"/mnt/data2/ML-Ready/GOES-18-SXR-B/{timestamp}.npy", sxr_b_data)

        return (timestamp, True, "Success")

    except Exception as e:
        return (timestamp, False, f"Error processing timestamp {timestamp}: {e}")


def update_progress(result):
    """Callback function to update progress bar"""
    global pbar, successful_count, failed_count
    timestamp, success, message = result

    if success:
        successful_count += 1
        pbar.set_postfix(success=successful_count, failed=failed_count)
    else:
        failed_count += 1
        pbar.set_postfix(success=successful_count, failed=failed_count)
        tqdm.write(f"Failed: {message}")

    pbar.update(1)


def main():
    global pbar, successful_count, failed_count

    # Collect timestamps found in each wavelength directory.
    timestamps_found = defaultdict(set)

    print("Scanning directories for timestamps...")
    for wavelength, dir_path in tqdm(wavelength_dirs.items(), desc="Scanning directories"):
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

    print(f"\nFound {len(common_timestamps)} timestamps present in all wavelength folders")
    print(f"Found {len(missing_files)} timestamps with missing wavelength files")

    # Load GOES data
    print("Loading GOES data...")
    goes = pd.read_csv("/mnt/data/goes_combined/combined_g18_avg1m_20230701_20230815.csv")
    goes['time'] = pd.to_datetime(goes['time'], format='%Y-%m-%d %H:%M:%S')

    # Create output directories if they don't exist
    os.makedirs("/mnt/data2/ML-Ready/AIA-Data", exist_ok=True)
    os.makedirs("/mnt/data2/ML-Ready/GOES-18-SXR-A", exist_ok=True)
    os.makedirs("/mnt/data2/ML-Ready/GOES-18-SXR-B", exist_ok=True)

    # Use all available CPU cores
    num_processes = cpu_count()
    print(f"Using {num_processes} CPU cores for processing")
    print(f"Processing {len(common_timestamps)} timestamps...")

    # Initialize global counters for progress tracking
    successful_count = 0
    failed_count = 0

    # Create arguments for multiprocessing (timestamp, goes_data pairs)
    args_list = [(timestamp, goes) for timestamp in common_timestamps]

    # Start timing
    start_time = time.time()

    # Create progress bar
    pbar = tqdm(total=len(common_timestamps), desc="Processing timestamps",
                unit="timestamp", dynamic_ncols=True)

    # Process timestamps in parallel with progress tracking
    with Pool(processes=num_processes) as pool:
        # Use map with callback for real-time progress updates
        results = []
        for args in args_list:
            result = pool.apply_async(process_timestamp, (args,), callback=update_progress)
            results.append(result)

        # Wait for all processes to complete
        for result in results:
            result.wait()

    # Close progress bar
    pbar.close()

    # Calculate statistics
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nProcessing complete!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per timestamp: {total_time / len(common_timestamps):.2f} seconds")
    print(f"Successfully processed: {successful_count}/{len(common_timestamps)} timestamps")
    print(f"Failed processes: {failed_count}")
    print(f"Processing rate: {len(common_timestamps) / total_time:.2f} timestamps/second")

    if failed_count > 0:
        print(f"\n{failed_count} timestamps failed processing (see messages above)")


if __name__ == "__main__":
    main()