import argparse
import glob
import logging
import os
from multiprocessing import Pool
import datetime
import dateutil.parser as dt
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from astropy.io.fits import getheader
from tqdm import tqdm
from sunpy.map import Map
from astropy import units as u


# Global variable
flare_class_mapping = {'A': 1e-8, 'B': 1e-7, 'C': 1e-6, 'M': 1e-5, 'X': 1e-4}


# Note: This function uses global variables.
def find_match(imager_date_delay):
    """Worker function for multithreading.

    Find matches for a given date.
    Parameters
    ----------
    imager_date_delay: date to match.

    Returns
    -------
    The match or None if no match was found.
    """
    imager_date, imager_delay = imager_date_delay
    nb_wl = len(imager_isodates)
    # Find min diff to aia observation
    eve_idx = np.argmin(np.abs(eve_dates - imager_delay))
    # Find the closest aia observation
    aia_ans = [min(imager_isodates[i], key=lambda sub: abs(sub - imager_date)) for i in range(nb_wl)]
    eve_ans = eve_dates[eve_idx]

    if abs(eve_ans - imager_delay) <= eve_to_imager_dt and np.amax(
            [abs(aia_ans[i] - imager_date) for i in range(nb_wl)]) <= datetime.timedelta(seconds=imager_dt):
        # save time difference between both observations
        time_delta = abs(eve_ans - imager_date)
        # get index from original array
        eve_index_match = eve_indices[eve_idx]
        eve_date_match = eve_dates[eve_idx]
        # get wavelength files
        imager_file_match = []
        for wl in range(nb_wl):
            # find index in array
            imager_idx = imager_isodates[wl].index(aia_ans[wl])
            imager_file = imager_filenames[wl][imager_idx]
            # TODO: Fix for STEREO (no quality flag)
            if getheader(imager_file, 1)['QUALITY'] != 0:
                LOG.error('Invalid quality flag encountered')
                return None
            imager_file_match += [imager_file]
        return (eve_date_match, eve_index_match, time_delta, imager_file_match, imager_date)
    return None


def _remove_flares(imager_matches, goes_data):
    """ Select imager dates and remove flaring times (>C4)


    Parameters
    ----------
    imager_matches: list of datetimes.
    goes_data: path to the GOES CSV file.

    Returns
    -------
    List of datetimes without flaring events.
    """
    flare_df = pd.read_csv(goes_data, parse_dates=['event_date', 'start_time', 'peak_time', 'end_time'])
    # parse goes classes to peak flux
    peak_flux = _to_goes_peak_flux(flare_df)
    # only flares >C4 have a significant impact on the full-disk images
    flare_df = flare_df[peak_flux >= 4e-6]

    # Create a mask to capture dates outside flaring times.  Uses the index of imager_matches, which is a date
    non_flare_mask = None    
    for i, row in flare_df.iterrows():
        if non_flare_mask is None:
            non_flare_mask = np.logical_or(imager_matches.index < row.start_time, imager_matches.index > row.end_time)
        else:
            temp_mask = np.logical_or(imager_matches.index < row.start_time, imager_matches.index > row.end_time)
            non_flare_mask = np.logical_and(non_flare_mask, temp_mask)

    # remove all dates during flaring times
    return imager_matches.loc[non_flare_mask, :]


def _load_imager_dates(filenames, one_hour_cad=False):
    """ load imager dates from filenames.

    Parameters
    ----------
    filenames: path names of the imager files.
    one_hour_cad: if True follow different naming convention.

    Returns
    -------
    List of imager datetimes.
    """
    if one_hour_cad:
        aia_dates = [[name.split(".")[-4][:-1] for name in wl_files] for wl_files in filenames]
    else:
        aia_dates = \
            [[name.split("/")[-1].split('.fits')[0].split('_A')[0].split('_B')[0].split("_")[-1] for name in wl_files]
             for wl_files in filenames]

    aia_iso_dates = [[dt.isoparse(date) for date in wl_dates] for wl_dates in aia_dates]
    return aia_iso_dates


def load_valid_eve_dates(eve_cdf, line_indices=None):
    """ load all valid dates from EVE

    Parameters
    ----------
    eve_cdf: NETCDF dataset of EVE.
    line_indices: wavelengths used for data check (optional).

    Returns
    -------
    numpy array of all valid EVE dates and corresponding indices
    """
    # load and parse eve dates
    eve_date_str = eve_cdf.variables['isoDate'][:]
    # convert to naive datetime object
    eve_dates = np.array([dt.isoparse(d).replace(tzinfo=None) for d in eve_date_str])
    # get all indices
    eve_indices = np.indices(eve_dates.shape)[0]
    # find invalid eve data points
    eve_data = eve.variables['irradiance'][:]
    if line_indices is not None:
        eve_data = eve_data[:, line_indices]
    # set -1 entries to nan
    eve_data[eve_data < 0] = np.nan
    # set outliers to nan
    # TODO: Check if condition still applies
    outlier_threshold = np.nanmedian(eve_data, 0) + 3 * np.nanstd(eve_data, 0)
    eve_data[eve_data > outlier_threshold[None]] = np.nan
    # filter eve dates and indices
    eve_dates = eve_dates[~np.any(np.isnan(eve_data), 1)]
    eve_indices = eve_indices[~np.any(np.isnan(eve_data), 1)]

    return eve_dates, eve_indices


def create_date_file_df(dates, files, prefix, dt_round='3min'):
    """ Parse a list of datetimes and files into dataframe

    Parameters
    ----------
    dates: list of dates
    files: list of filepaths
    prefix: string to use in the creation of the columns of the df. Typically, an
            AIA wavelength or the name 'hmi'
    dt_round: frequency alias to round dates
        see https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.round.html


    Returns
    -------
    pandas df with datetime index and paths
    """
    df1 = pd.DataFrame(data={'dates': dates, f'{prefix}': files, f'dates_{prefix}': dates})
    df1['dates'] = df1['dates'].dt.round(dt_round)
    # Drop duplicates in case datetimes round to the same value
    df1 = df1.drop_duplicates(subset='dates', keep='first')
    df1 = df1.set_index('dates', drop=True)

    return df1


def match_imager_times(all_iso_dates, all_filenames, all_prefixes, joint_df=None, dt_round='3min'):
    """ Parses aia_iso_dates and compile lists at the same time

    Parameters
    ----------
    all_iso_dates: list of AIA channel datetimes
    all_filenames: filenames of AIA files
    all_prefixes: list of strings to use in the creation of the columns of the df. Typically, AIA wavelengths or name
    joint_df: pandas dataframe to use as a starting point
    dt_round: frequency alias to round dates
        see https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.round.html

    Returns
    -------
    pandas dataframe of matching datetimes
    """

    # Create a dataframe with all dates
    date_columns = []
    # Create a list of columns to keep
    columns_to_keep = ['median_dates']
    # Loop through all AIA channels
    for n, (aia_iso_dates, aia_filenames, prefix) in enumerate(zip(all_iso_dates, all_filenames, all_prefixes)):
        date_columns.append(f'dates_{prefix}')
        columns_to_keep.append(f'{prefix}')
        df = create_date_file_df(aia_iso_dates, aia_filenames, prefix)
        # Join the dataframes
        if n == 0 and joint_df is None:
            joint_df = df
        else:
            joint_df = joint_df.join(df, how='inner')

    # Find the median date
    joint_df['median_dates'] = joint_df.loc[:, date_columns].median(numeric_only=False, axis=1)
    # Drop all columns that are not needed
    joint_df = joint_df.loc[:, columns_to_keep]

    return joint_df


def _to_goes_peak_flux(flare_df):
    """Parses goes class (str) to peak flux (float)

    Parameters
    ----------
    flare_df: GOES data frame.

    Returns
    -------
    Numpy array with the peak flux values.
    """
    peak_flux = [flare_class_mapping[fc[0]] * float(fc[1:]) for fc in flare_df.goes_class]
    return np.array(peak_flux)


def str2bool(v):
    """Convert string to boolean.

    Parameters
    ----------
    v: string to convert to boolean.

    Returns
    -------
    Boolean value.
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    """Main function to find matches between EUV imagers and irradiance data.
    
    Parameters
    ----------
    -imager: list of EUV imagers.
    -imager_dir: list of directories containing EUV imager data.
    -imager_wl: list of wavelengths to match.
    -imager_dt: time delta between EUV images in different wavelengths.
    -eve_data: path to EVE csv file.
    -goes_data: path to GOES csv file.
    -output_path: path to store output csv file.
    -eve_to_imager_dt: time delta between EVE and EUV images.
    -time_delay: account for time delay between EUV observations and irradiance measurements.
    
    Returns
    -------
    CSV file containing matches between EUV imagers and irradiance data.
    """

    # Logging
    logging.basicConfig(format='%(levelname)-4s '
                               '[%(module)s:%(funcName)s:%(lineno)d]'
                               ' %(message)s')
    LOG = logging.getLogger()
    LOG.setLevel(logging.INFO)

    # Parser
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-imager', type=str, nargs='+', default=['AIA'], help='EUV imagers.')  
    p.add_argument('-imager_dir', type=str, nargs="+", 
                   default=["/mnt/disks/data-spi3s-irradiance/AIA"],
                   help='Path to directory of aia files')
    p.add_argument('-imager_wl', type=str, nargs="+",
                   default=["94", "131", "171", "193", "211", "304", "335", "1600", "1700"],
                   help='List of wavelengths that are needed')
    p.add_argument('-imager_dt', type=int, default=600,
                   help='Cutoff for time delta (difference between AIA images in different wavelengths) in seconds')
    p.add_argument('-eve_data', type=str,
                   default="/mnt/disks/data-spi3s-irradiance/EVE/EVS_MEGS-AB_irradiance.nc",
                   help='Path to netCDF file')
    p.add_argument('-goes_data', type=str, default="/mnt/disks/data-spi3s-irradiance/GOES/goes.csv",
                   help='Path to GOES csv file')
    p.add_argument('-output_path', type=str,
                   default="/mnt/disks/data-spi3s-irradiance/matches/matches_EVS_AIA.csv",
                   help='Path to store output csv file')
    p.add_argument('-eve_to_imager_dt', type=int, default=600,
                   help='Cutoff for time delta (difference between AIA and EVE file in time) in seconds')
    p.add_argument('-time_delay', type=str2bool, default=False,
                   help='Account for time delay between EUV observations and irradiance measurements.')
    args = p.parse_args()

    # Parse arguments
    imager = args.imager
    nb_imager = len(imager)
    eve_data = args.eve_data
    goes_data = args.goes_data
    output_path = args.output_path
    imager_wl = args.imager_wl
    eve_to_imager_dt = args.eve_to_imager_dt
    imager_dt = args.imager_dt
    time_delay = args.time_delay

    # Load EVE cdf file
    eve = Dataset(eve_data, "r", format="NETCDF4")
    eve_dates, eve_indices = load_valid_eve_dates(eve)
    eve_to_imager_dt = datetime.timedelta(seconds=eve_to_imager_dt)

    # Imager variables: filenames, prefixes, and time delta
    imager_filenames = []
    imager_prefixes = []
    
    # Loop through EUV imagers and match wavelengths
    for i, inst in enumerate(imager):
        # Find directory containing imager data
        imager_dir = args.imager_dir[i]
        # Loops through subdirectories to find all available wavelengths
        found_wl = sorted([d for d in os.listdir(imager_dir) if os.path.isdir(imager_dir+'/'+d)])
        # Find intersection between available wavelengths and requested wavelengths
        intersection_wl = [str(wl) for wl in sorted([int(wl) for wl in list(set(found_wl).intersection(imager_wl))])]
        # For matching wavelengths, add prefixes to list, and filenames to list
        imager_prefixes.extend([f'{inst}{wl}' for wl in intersection_wl])
        filenames = [[f for f in sorted(glob.glob(imager_dir + f'/%s/*.fits' % wl))] for wl in intersection_wl]
        # Extend list of imager filenames
        imager_filenames.extend(filenames)    
    
    # Find matches in time between images from all EUV imagers 
    imager_isodates = _load_imager_dates(imager_filenames)    
    imager_matches = match_imager_times(imager_isodates, imager_filenames, imager_prefixes, dt_round=f'{imager_dt}s')
    # Filter out solar flares
    imager_matches = _remove_flares(imager_matches, goes_data)

    # Apply time delay between EUV observations and irradiance measurements
    # Convert dates to datetime64 format
    imager_matched_dates = imager_matches['median_dates'].to_numpy().astype('datetime64[us]').tolist()
    imager_delayed_dates = []
    # Calculate time delay for each EUV observation
    if time_delay:
        # Rotational frequency of the Sun
        rot_freq = 360 / 27.26
        # Calculate time delay for each EUV observation
        dt_delay = []
        for i, filename in enumerate(imager_matches['B/EUVI171']):
            # Read Sunpy map to get heliographic longitude
            s_map = Map(filename)
            lon = s_map.heliographic_longitude.to(u.deg).value
            dt_delay.append(datetime.timedelta(days=-lon / rot_freq))
            imager_delayed_dates.append(imager_matched_dates[i]+datetime.timedelta(days=-lon / rot_freq))
        imager_matches['eve_delay'] = dt_delay
    else:
        imager_delayed_dates = imager_matched_dates
    
    # looping through AIA filenames to find matching EVE files
    nb_workers = os.cpu_count()
    with Pool(nb_workers) as p:
        # Find matches in time between imager and irradiance data
        matches = [result for result in tqdm(p.imap(find_match, zip(imager_matched_dates, imager_delayed_dates)),
                                             total=len(imager_matched_dates)) if result is not None]

    # Unpack and create result dataframe --> Save as CSV
    if time_delay is True and imager == ['B/EUVI']:
        result_matches = pd.DataFrame({"eve_dates": [eve_dates for eve_dates, _, _, _, _ in matches],
                                       "eve_indices": [eve_indices for _, eve_indices, _, _, _ in matches],
                                       "time_delta": [time_delta for _, _, time_delta, _, _ in matches], 
                                       "dates": [date for _, _, _, _, date in matches]})
        result_matches = result_matches.set_index('dates', drop=True)
        result_matches = result_matches.join(imager_matches, how='inner')
        result_matches = result_matches.set_index('eve_dates', drop=False)
    else: 
        result_matches = pd.DataFrame({"eve_dates": [eve_dates for eve_dates, _, _, _, _ in matches],
                                       "eve_indices": [eve_indices for _, eve_indices, _, _, _ in matches],
                                       "time_delta": [time_delta for _, _, time_delta, _, _ in matches]})
        result_matches['dates'] = result_matches['eve_dates'].dt.round(f'{imager_dt}s')
        result_matches = result_matches.set_index('dates', drop=True)
        result_matches = result_matches.join(imager_matches, how='inner')
    
    # Save csv with aia filenames, aia iso dates, eve iso dates, eve indices, and time deltas
    if not os.path.exists(output_path[:output_path.rfind("/")]):
        os.makedirs(output_path[:output_path.rfind("/")], exist_ok=True)
    result_matches.to_csv(output_path, index=False)
