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

        logging.info(f"Found {len(goes_query[0])} GOES files.")

        # Skip if no files found
        if len(goes_query[0]) == 0:
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

    def combine_goes_data(self, columns_to_interp=["xrsb_flux", "xrsa_flux"]):
        """
        Combine GOES-16 and GOES-18 files and track source files used.
        Parameters
        """

        g13_files = sorted(self.save_dir.glob("*g13*.nc"))
        g14_files = sorted(self.save_dir.glob("*g14*.nc"))
        g15_files = sorted(self.save_dir.glob("*g15*.nc"))
        g16_files = sorted(self.save_dir.glob("*g16*.nc"))
        g17_files = sorted(self.save_dir.glob("*g17*.nc"))
        g18_files = sorted(self.save_dir.glob("*g18*.nc"))
        logging.info(
            f"Found {len(g13_files)} GOES-13 files, {len(g14_files)} GOES-14 files, {len(g15_files)} GOES-15 files, {len(g16_files)} GOES-16 files, {len(g17_files)} GOES-17 files, and {len(g18_files)} GOES-18 files.")

        def process_files(files, satellite_name, output_file, used_file_list):
            datasets = []
            combined_meta = {}

            for file_path in files:
                try:
                    ds = xr.open_dataset(str(file_path))
                    datasets.append(ds)
                    used_file_list.append(file_path)  # Track file used
                    logging.info(f"Loaded {file_path.name}")
                except Exception as e:
                    logging.error(f"Could not load {file_path.name}: {e}")
                    continue
                finally:
                    if 'ds' in locals():
                        ds.close()

            if not datasets:
                logging.warning(f"No valid datasets for {satellite_name}")
                return

            try:
                combined_ds = xr.concat(datasets, dim='time').sortby('time')
                if satellite_name in ['GOES-13', 'GOES-14', 'GOES-15']:
                    combined_ds['xrsa_flux'] = combined_ds['xrsa_flux'] / .85
                    combined_ds['xrsb_flux'] = combined_ds['xrsb_flux'] / .7
                df = combined_ds.to_dataframe().reset_index()
                if 'quad_diode' in df.columns:
                    df = df[df['quad_diode'] == 0]  # Filter out quad diode data
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df_log = np.log10(df[columns_to_interp].replace(0, np.nan))

                # Step 3: Interpolate in log space
                df_log_interp = df_log.interpolate(method="time", limit_direction="both")

                # Step 4: Back-transform to linear space
                df[columns_to_interp] = 10 ** df_log_interp

                # Add min and max dates to filename
                min_date = df.index.min().strftime('%Y%m%d')
                max_date = df.index.max().strftime('%Y%m%d')
                filename = f"{str(output_file)}_{min_date}_{max_date}.csv"
                df.to_csv(filename, index=False)

                logging.info(f"Saved combined file: {output_file}")
            except Exception as e:
                logging.error(f"Failed to write {output_file}: {e}")
            finally:
                for ds in datasets:
                    ds.close()

        if len(g13_files) != 0:
            process_files(g13_files, "GOES-13", self.concat_dir / "combined_g13_avg1m", self.used_g13_files)
        if len(g14_files) != 0:
            process_files(g14_files, "GOES-14", self.concat_dir / "combined_g14_avg1m", self.used_g14_files)
        if len(g15_files) != 0:
            process_files(g15_files, "GOES-15", self.concat_dir / "combined_g15_avg1m", self.used_g15_files)
        if len(g16_files) != 0:
            process_files(g16_files, "GOES-16", self.concat_dir / "combined_g16_avg1m", self.used_g16_files)
        if len(g17_files) != 0:
            process_files(g17_files, "GOES-17", self.concat_dir / "combined_g17_avg1m", self.used_g17_files)
        if len(g18_files) != 0:
            process_files(g18_files, "GOES-18", self.concat_dir / "combined_g18_avg1m", self.used_g18_files)