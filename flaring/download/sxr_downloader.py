from pathlib import Path
import numpy as np
from sunpy.net import Fido, attrs as a
from concurrent.futures import ThreadPoolExecutor
import logging
import xarray as xr
import pandas as pd


class SXRDownloader:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, save_dir: str = '/downloads/goes_data', concat_dir: str = '/downloads/goes_combined'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.concat_dir = Path(concat_dir)
        self.concat_dir.mkdir(exist_ok=True)
        self.used_g13_files = []
        self.used_g14_files = []
        self.used_g15_files = []
        self.used_g16_files = []
        self.used_g17_files = []
        self.used_g18_files = []

    def download_and_save_goes_data(self, start='2023-07-01', end='2023-08-15', max_workers=4):
        """
        Download GOES X-ray data at 1-minute cadence for a specified time range, including all available satellites.

        Parameters:
        - start (str): Start date for the query (e.g., '2023-07-01')
        - end (str): End date for the query (e.g., '2023-08-15')
        - max_workers (int): Number of parallel download threads
        """
        logging.info(f"Searching GOES X-ray data from {start} to {end} for all satellites at 1-minute cadence...")

        # Query for all GOES satellites with 1-minute averaged XRS data
        goes_query = Fido.search(
            a.Time(start, end),
            a.Instrument('XRS'),
            a.Resolution('avg1m')  # 1-minute averaged data
        )

        logging.info(f"Found {len(goes_query)} GOES files.")

        # Skip if no files found
        if len(goes_query) == 0:
            logging.warning("No files found for the specified query.")
            return []

        # Define download function for a single file
        def download_file(file_entry, path_template):
            try:
                fido_result = Fido.fetch(file_entry, path=str(path_template / "{file}"))
                return fido_result
            except Exception as e:
                logging.error(f"Failed to download {file_entry['file']}: {e}")
                return []

        # Use ThreadPoolExecutor for parallel downloads
        logging.info(f"Downloading {len(goes_query[0])} files with {max_workers} workers...")
        downloaded_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create download tasks for each file
            futures = [
                executor.submit(download_file, row, self.save_dir)
                for row in goes_query[0]
            ]
            # Collect results
            for future in futures:
                result = future.result()
                downloaded_files.extend(result)

        logging.info(f"Saved {len(downloaded_files)} files to {self.save_dir}")
        return downloaded_files