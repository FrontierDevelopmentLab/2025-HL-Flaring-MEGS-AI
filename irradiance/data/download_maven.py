import os
import argparse
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import wget
from tqdm import tqdm
from irradiance.data.download_eve import query_url


def datetime_to_format(date_str):
    """Convert datetime object to EVE filename format.

    Parameters
    ----------
    date_str: str. Date in format YYYY-MM-DDTHH:MM:SS to convert
              to datetime object and then to EVE filename format.

    Returns
    -------
    tuple. Year, day of year, hour, minute.
    """
    dt = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
    return dt.year, dt.month, dt.day


if __name__ == "__main__":
    """Download EVE data from LASP website within a user-defined time range.

    Parameters
    ----------
    start: str. Time to start downloading data in format YYYY-MM-DDTHH:MM:SS.
    end: str. Time to stop downloading data in format YYYY-MM-DDTHH:MM:SS.
    type: str. Data type: EVL (lines or bands) and EVS (spectra).
    level: str. Data level: 0, 1, 2, 2B, 3, 4.
    version: str. Data version. Currently only version 8 is available.
    url: str. Website URL to download data from.
    save_dir: str. Local path to save data to.

    Returns
    -------
    list of folders and files (shape: n_folders, n_files)
    Download irradiance data files from the EVE website.
    """

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', type=str, default='2010-01-01T00:00:00',
                        help='Enter start time for data downloading in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('-end', type=str, default='2015-01-01T00:00:00',
                        help='Enter end time for data downloading in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('-level', type=str, default='2',
                        help='Specify data level: 2, 3')
    parser.add_argument('-url', type=str,
                        default='https://lasp.colorado.edu/maven/sdc/public/data/sci/euv',
                        help='Specify data download url')
    parser.add_argument('-save_dir', type=str,
                        default='/mnt/disks/data-spi3s-irradiance/EVE',
                        help='Specify where to store data')
    args = parser.parse_args()

    # Pass arguments to variables
    start_date = args.start
    end_date = args.end
    data_level = args.level
    data_url = os.path.join(args.url, f'l{args.level}')
    data_save_dir = args.save_dir

    # Create save_dir if it does not exist
    os.makedirs(data_save_dir, exist_ok=True)

    # Start & end times in EVE format
    start_year, start_month, start_day = datetime_to_format(start_date)
    end_year, end_month, end_day = datetime_to_format(end_date)

    # Pulling available years from EVE website
    year_query = query_url(data_url, f'^2')
    # Constrain query within the start and end years
    year_query = [y for y in year_query if int(start_year) <= int(y) <= int(end_year)]
    print(year_query)

    # Loop over years
    for year in tqdm(year_query, desc='Years'):

        # Pulling available days for a given year
        months_query = query_url(os.path.join(data_url, year), f'^{np.arange(0, 2)}')
        print(months_query)
        # Constrain query within the start and end years and corresponding days
        if int(year) == int(start_year):
            months_query = [d for d in months_query if int(d) >= int(start_month)]
        if int(year) == int(end_year):
            months_query = [d for d in months_query if int(d) <= int(end_month)]

            # Loop over days
        for month in tqdm(months_query, desc='Days'):

            # Pulling available fits files for a given day
            fits_query = query_url(os.path.join(data_url, year, month), f'.cdf$')
            fits_query = [f for f in fits_query if int(f"{start_year}{start_month:02d}{start_day:02d}") <=
                          int(f.split('_')[4]) <= int(f"{end_year}{end_month:02d}{end_day:02d}")]

            # Loop over fits files
            for data in fits_query:

                # Download if file does not exist
                if os.path.exists(os.path.join(data_save_dir, data)):
                    print(f'{data} already exists, skipping...')
                else:
                    wget.download(os.path.join(data_url, year, month, data), os.path.join(data_save_dir, data))
