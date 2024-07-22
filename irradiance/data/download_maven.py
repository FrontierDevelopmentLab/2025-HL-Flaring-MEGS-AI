import os
import argparse
import requests
import re
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
import wget
from tqdm import tqdm


def datetime_to_eve_format(date_str):
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
    return dt.year, dt.timetuple().tm_yday, dt.hour, dt.minute


def query_url(url, pattern):
    """Query a URL and return the HTML content. Parse text for data.

    Parameters
    ----------
    url: str. URL to query.

    Returns
    -------
    List. Relevant content of the URL.
    """
    result = requests.get(url)
    soup = BeautifulSoup(result.text, 'html.parser')
    query = [a.get('href').replace('/', '') for a in soup.find_all(href=re.compile(pattern))]
    return query


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
    parser.add_argument('-type', type=str, default='EVS',
                        help='Specify data type: EVL (lines or bands) and EVS (spectra)')
    parser.add_argument('-level', type=str, default='2B',
                        help='Specify data level: 0, 1, 2, 2B, 3, 4')
    parser.add_argument('-version', type=str, default='008',
                        help='Specify data version')
    parser.add_argument('-url', type=str,
                        default='https://lasp.colorado.edu/eve/data_access/eve_data/products/',
                        help='Specify data download url')
    parser.add_argument('-save_dir', type=str,
                        default='/mnt/disks/data-spi3s-irradiance/EVE',
                        help='Specify where to store data')
    args = parser.parse_args()

    # Pass arguments to variables
    start_date = args.start
    end_date = args.end
    data_type = args.type
    data_version = args.version
    data_url = os.path.join(args.url, f'level{args.level}')
    data_save_dir = args.save_dir

    # Create save_dir if it does not exist
    os.makedirs(data_save_dir, exist_ok=True)

    # Start & end times in EVE format
    start_year, start_yday, start_hour, start_min = datetime_to_eve_format(start_date)
    end_year, end_yday, end_hour, end_min = datetime_to_eve_format(end_date)

    # Pulling available years from EVE website
    year_query = query_url(data_url, f'^2')
    # Constrain query within the start and end years
    year_query = [y for y in year_query if int(start_year) <= int(y) <= int(end_year)]

    # Loop over years
    for year in tqdm(year_query, desc='Years'):

        # Pulling available days for a given year
        days_query = query_url(os.path.join(data_url, year), f'^{np.arange(0, 4)}')
        # Constrain query within the start and end years and corresponding days
        if int(year) == int(start_year):
            days_query = [d for d in days_query if int(d) >= int(start_yday)]
        if int(year) == int(end_year):
            days_query = [d for d in days_query if int(d) <= int(end_yday)]

            # Loop over days
        for day in tqdm(days_query, desc='Days'):

            # Pulling available fits files for a given day
            fits_query = query_url(os.path.join(data_url, year, day), f'^{data_type}')
            # Constrain query within the start and end years, days and corresponding hours
            if int(year) == int(start_year) and int(day) == int(start_yday):
                fits_query = [f for f in fits_query if int(f.split('_')[3]) >= int(start_hour)]
            if int(year) == int(end_year) and int(day) == int(end_yday):
                fits_query = [f for f in fits_query if int(f.split('_')[3]) <= int(end_hour)]

            # Loop over fits files
            for data in fits_query:

                # Download if file does not exist
                if os.path.exists(os.path.join(data_save_dir, data)):
                    print(f'{data} already exists, skipping...')
                else:
                    wget.download(os.path.join(data_url, year, day, data), os.path.join(data_save_dir, data))
