import argparse
import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset


# TODO: Adapt for both EVS and EVL data
def _load(eve_nc_path, matches_csv, output_path, output_stats, output_wl):
    """ Load EVE data, select matches and save as numpy file.

    Avoid loading the full dataset for each model training.
    Parameters
    ----------
    eve_nc_path: path to the NETCDF file.
    matches_csv: path to the CSV matches file.
    output_path: output path for the numpy files.

    Returns
    -------
    None
    """
    line_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14])

    # load eve data
    eve_data_db = Dataset(eve_nc_path)
    eve_data = eve_data_db.variables['irradiance'][:]
    # eve_data = eve_data[:, line_indices]

    # load matches
    matches_csv = pd.read_csv(matches_csv)
    eve_data = eve_data[matches_csv.eve_indices]

    # normalize data between 0 and max
    eve_mean = np.nanmean(eve_data, 0)
    eve_std = np.nanstd(eve_data, 0)
    # eve_data = (eve_data - eve_mean[None]) / eve_std[None]

    # save eve data
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    np.save(output_path, eve_data.data)

    # save normalization
    stats = (eve_mean, eve_std)
    np.save(output_stats, stats)

    # save wl names
    # names = eve_data_db.variables['name'][:][line_indices]
    # np.save(output_wl, names.data)

    eve_data_db.close()


if __name__ == "__main__":
    """ Load EVE data, select matches and save as numpy file.
    
    Parameters
    ----------
    eve_data: path to the NETCDF file.
    matches_csv: path to the CSV matches file.
    output_data: output path for the numpy files.
    output_stats: output path for the numpy files.
    output_wl: output path for the wavelength names.
    
    Returns
    -------
    None. Stores data in the specified paths.
    
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-eve_data', type=str,
                   default="/mnt/disks/data-spi3s-irradiance/EVE/EVS_MEGS-AB_irradiance.nc",
                   help='path to the NETCDF file.')
    p.add_argument('-matches_csv', type=str,
                   default="/mnt/disks/data-spi3s-irradiance/matches/EVS_MEGS-AB_matches.csv",
                   help='path to the CSV matches file.')
    p.add_argument('-output_data', type=str,
                   default="/mnt/disks/data-spi3s-irradiance/matches/EVE/EVS_MEGS-AB_irradiance.npy",
                   help='output path for the numpy files.')
    p.add_argument('-output_stats', type=str,
                   default="/mnt/disks/data-spi3s-irradiance/matches/EVE/EVS_MEGS-AB_stats.npz",
                  help='output path for the numpy files.')
    p.add_argument('-output_wl', type=str,
                   default=None,
                   help='output path for the wavelength names.')
    args = p.parse_args()

    # load and save data
    _load(args.eve_data, args.matches_csv, args.output_data, args.output_stats, args.output_wl)
