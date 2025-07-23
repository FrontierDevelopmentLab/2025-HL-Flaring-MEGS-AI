import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class SXRDataProcessor:
    """Class to process GOES X-ray data, including downloading, combining, and interpolating data from multiple satellites.
    This class handles the downloading of GOES data, combining data from different satellites, and applying interpolation
    in log space to the X-ray flux data.
    It also tracks which files were used in the processing.
    Parameters
    ----------
    save_dir : str
        Directory where downloaded GOES data will be saved.
    concat_dir : str
        Directory where combined GOES data will be saved.
    """

    def __init__(self, data_dir: str = '/mnt/data/downloads/goes_data', output_dir: str = '/downloads/goes_combined'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.used_g13_files = []
        self.used_g14_files = []
        self.used_g15_files = []
        self.used_g16_files = []
        self.used_g17_files = []
        self.used_g18_files = []

    def combine_goes_data(self, columns_to_interp=["xrsb_flux", "xrsa_flux"]):
        """
        Combine GOES-16 and GOES-18 files and track source files used.
        Parameters
        """

        g13_files = sorted(self.data_dir.glob("*g13*.nc"))
        g14_files = sorted(self.data_dir.glob("*g14*.nc"))
        g15_files = sorted(self.data_dir.glob("*g15*.nc"))
        g16_files = sorted(self.data_dir.glob("*g16*.nc"))
        g17_files = sorted(self.data_dir.glob("*g17*.nc"))
        g18_files = sorted(self.data_dir.glob("*g18*.nc"))
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
                # Scaling factors for GOES-13, GOES-14, and GOES-15
                if satellite_name in ['GOES-13', 'GOES-14', 'GOES-15']:
                    combined_ds['xrsa_flux'] = combined_ds['xrsa_flux'] / .85
                    combined_ds['xrsb_flux'] = combined_ds['xrsb_flux'] / .7
                df = combined_ds.to_dataframe().reset_index()
                #
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
                df.to_csv(filename, index=True)

                logging.info(f"Saved combined file: {output_file}")
            except Exception as e:
                logging.error(f"Failed to write {output_file}: {e}")
            finally:
                for ds in datasets:
                    ds.close()

        if len(g13_files) != 0:
            process_files(g13_files, "GOES-13", self.output_dir / "combined_g13_avg1m", self.used_g13_files)
        if len(g14_files) != 0:
            process_files(g14_files, "GOES-14", self.output_dir / "combined_g14_avg1m", self.used_g14_files)
        if len(g15_files) != 0:
            process_files(g15_files, "GOES-15", self.output_dir / "combined_g15_avg1m", self.used_g15_files)
        if len(g16_files) != 0:
            process_files(g16_files, "GOES-16", self.output_dir / "combined_g16_avg1m", self.used_g16_files)
        if len(g17_files) != 0:
            process_files(g17_files, "GOES-17", self.output_dir / "combined_g17_avg1m", self.used_g17_files)
        if len(g18_files) != 0:
            process_files(g18_files, "GOES-18", self.output_dir / "combined_g18_avg1m", self.used_g18_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess GOES X-ray data.')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/downloads/goes_data',
                        help='Directory where downloaded GOES data is stored.')
    parser.add_argument('--output_dir', type=str, default='/mnt/data/downloads/goes_combined',
                        help='Directory where combined GOES data will be saved.')
    args = parser.parse_args()
    processor = SXRDataProcessor(data_dir=args.data_dir, output_dir=args.output_dir)
    processor.combine_goes_data()

    print("GOES data processing completed.")
