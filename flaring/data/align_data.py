import glob
import os
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')


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
        sxr_a_data = np.zeros(1, dtype=np.float32)
        sxr_b_data = np.zeros(1, dtype=np.float32)
        sxr_a_data[0] = sxr_a
        sxr_b_data[0] = sxr_b

        # Save data to disk
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

    # Load GOES data
    print("Loading GOES data...")
    goes = pd.read_csv("/mnt/data2/goes_combined/combined_g18_avg1m_20230701_20230815.csv")
    goes['time'] = pd.to_datetime(goes['time'], format='%Y-%m-%d %H:%M:%S')

    # Create output directories if they don't exist
    os.makedirs("/mnt/data2/ML-Ready/GOES-18-SXR-A", exist_ok=True)
    os.makedirs("/mnt/data2/ML-Ready/GOES-18-SXR-B", exist_ok=True)

    aia_files = sorted(glob.glob('/mnt/data2/AIA_processed/*.npy', recursive=True))
    aia_files_split = []
    for file in aia_files:
        aia_files_split.append(file.split('/')[4].split('.')[0])

    common_timestamps = [
        datetime.fromisoformat(date_str).strftime('%Y-%m-%d %H:%M:%S')
        for date_str in aia_files_split]

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
