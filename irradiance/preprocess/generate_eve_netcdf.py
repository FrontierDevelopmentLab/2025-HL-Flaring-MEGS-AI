import numpy as np
import dateutil.parser as dt
from tqdm import tqdm
import argparse
from netCDF4 import Dataset
import os
import sys
import inspect
import glob
# Add 2024-HL-SPI3S/spi3s as a module
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))), '2024-HL-SPI3S')
sys.path.insert(0, parentdir)
from spi3s.data.eve_download import datetime_to_eve_format
from spi3s.data.eve_read import read_fits, read_evs, fits_to_df


def create_eve_netcdf(netcdf_outpath, eve_irr, eve_date, eve_wl):
    """_summary_

    Args:
        eve_raw_path (str): path to directory of eve files
        netcdf_outpath (str): path to output cdf file

    Returns:
        netcdf file containing EVE data and metadata
    """
    os.makedirs(os.path.split(netcdf_outpath)[0], exist_ok=True)

    ####################################################################################################################
    # CREATE NETCDF4 FILE
    ####################################################################################################################
    netcdfDB = Dataset(netcdf_outpath, "w", format="NETCDF4")
    netcdfDB.title = 'EVE spectral irradiance for specific spectral lines'

    # Create dimensions
    isoDate = netcdfDB.createDimension("isoDate", None)
    name = netcdfDB.createDimension("name", eve_wl.shape[0])
    # name = netcdfDB.createDimension("name", eve_name.shape[0])

    # Create variables and atributes
    isoDates = netcdfDB.createVariable('isoDate', 'S2', ('isoDate',))
    isoDates.units = 'string date in ISO format'

    # names = netcdfDB.createVariable('name', 'S2', ('name',))
    # names.units = 'strings with the line names'

    wavelength = netcdfDB.createVariable('wavelength', 'f4', ('name',))
    wavelength.units = 'line wavelength in nm'

    # logt = netcdfDB.createVariable('logt', 'f4', ('name',))
    # logt.units = 'log10 of the temperature'

    irradiance = netcdfDB.createVariable('irradiance', 'f4', ('isoDate', 'name',))
    irradiance.units = 'spectral irradiance in the specific line (w/m^2)'

    # Initialize variables
    isoDates[:] = eve_date
    # julianDates[:] = eve_jd
    # names[:] = eve_name
    wavelength[:] = eve_wl
    # logt[:] = eve_logt
    irradiance[:] = eve_irr

    netcdfDB.close()


def fits_to_arr(data, wl, wl_start, wl_end, t):
    """Convert FITS data to pandas dataframe.

    Parameters
    ----------
    data: Irradiance data from FITS file.
    wl: Wavelengths of spectra.
    wl_start: Start wavelength.
    wl_end: End wavelength.
    t: Time of spectra.
    flags: Binary flags to identify the quality of the data.

    Returns
    -------
    df: Pandas dataframe with irradiance data. Rows = Time, Columns = Wavelengths.
    """

    # Wavelength filtering
    wl_where = np.where((wl > wl_start) & (wl < wl_end))
    wl_filtered = wl[wl_where[0]]
    data_filtered = data[:, wl_where[0]]
    # flags_filtered = flags[:, wl_where[0]]

    # Time filtering
    # t_where = np.where(np.sum(flags_filtered, axis=1) == 0)
    t_where = np.where(np.sum(data_filtered, axis=1) > 0)
    data_filtered = data_filtered[t_where[0]]
    t_filtered = t[t_where[0]]

    # Return filtered data
    return data_filtered, wl_filtered, t_filtered


if __name__ == "__main__":
    """Read all EVE_V8 EVS FITS files and return store irradiance into dataframes.

        Parameters
        ----------
        start: str. Time to start downloading data in format YYYY-MM-DDTHH:MM:SS.
        end: str. Time to stop downloading data in format YYYY-MM-DDTHH:MM:SS.
        type: str. Data type: EVL (lines or bands) and EVS (spectra).
        level: str. Data level: 0, 1, 2, 2B, 3, 4.
        version: str. Data version. Currently only version 8 is available.
        datapath: str. Local path to EVS data.
        savepath: str. Local path to save data to.

        Returns
        -------
        df_megsa: Pandas dataframe with MEGS-A data.
        df_megsb: Pandas dataframe with MEGS-B data.
        df_megsab: Pandas dataframe with MEGS-A and MEGS-B data.
        
        Notes:
        ------
        Level 2 spectra are the merged spectral measurements from the two spectrographs, MEGS-A and MEGS-B. The
        A detector is designed to measure from 6 –17 nm, and 17–37 nm using two filters, while the B detector is
        designed to measure 37–106. After the MEGS-A anomaly, MEGS-B was extended down to 33.33 nm.
        Source: https://lasp.colorado.edu/eve/data_access/eve_data/products/level2/EVE_L2_V8_README.pdf
    """
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', type=str, required=True,
                        help='Enter start time for data downloading in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('-end', type=str, required=True,
                        help='Enter end time for data downloading in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('-type', type=str, default='EVS',
                        help='Specify data type: EVL (lines or bands) and EVS (spectra)')
    parser.add_argument('-level', type=str, default='2B',
                        help='Specify data level: 0, 1, 2, 2B, 3, 4')
    parser.add_argument('-version', type=str, default='008',
                        help='Specify data version')
    parser.add_argument('-revision', type=str, default='01',
                        help='Specify data revision number')
    parser.add_argument('-path', type=str,
                        default='/mnt/disks/data-spi3s-irradiance/EVE',
                        help='Specify where data is stored locally')
    parser.add_argument('-savepath', type=str,
                        default='/mnt/disks/data-spi3s-irradiance/EVE',
                        help='Specify where to store data')
    args = parser.parse_args()

    # Pass arguments to variables
    start_date = args.start
    end_date = args.end
    data_type = args.type
    data_level = args.level
    data_version = args.version
    data_revision = args.revision
    datapath = args.path
    savepath = args.savepath

    # Wavelengths - MEGS-A
    wl_megsa_start, wl_megsa_end = 5.77, 33.33
    # Wavelengths - MEGS-B
    wl_megsb_start, wl_megsb_end = wl_megsa_end, 106.61

    # Start & end times in EVE format
    start_year, start_yday, start_hour, start_min = datetime_to_eve_format(start_date)
    end_year, end_yday, end_hour, end_min = datetime_to_eve_format(end_date)

    # Files
    evs_files = sorted(glob.glob(os.path.join(datapath, f"{data_type}_L{data_level}_*_{data_version}_{data_revision}.fit.gz")))
    # Extract files between start time and end time
    evs_files = [e for e in evs_files if 1000*start_year+start_yday <= int(e.split('_')[2]) <= 1000*end_year+end_yday]

    first_megsa = True
    first_megsb = True
    first_megsab = True

    for i, evs in tqdm(enumerate(evs_files), total=len(evs_files)):
        # Read EVS file: Wavelengths, Time, Irradiance, and Binary Flags
        wl, irr_t, irr_data, bin_flags = read_evs(evs, verbose=False)
        # Dimensions
        nb_wl, nb_t = len(wl), len(irr_t)

        # Extract MEGS-A and MEGS-B data as a function of time and wavelengths
        # df_megsa_i = fits_to_df(irr_data, wl, wl_megsa_start, wl_megsa_end, irr_t, bin_flags)
        # df_megsb_i = fits_to_df(irr_data, wl, wl_megsb_start, wl_megsb_end, irr_t, bin_flags)

        # Extract MEGS-A and MEGS-B data as a function of time and wavelengths
        irr_megsa_i, wl_megsa, t_megsa_i = fits_to_arr(irr_data, wl, wl_megsa_start, wl_megsa_end, irr_t)
        irr_megsb_i, wl_megsb, t_megsb_i = fits_to_arr(irr_data, wl, wl_megsb_start, wl_megsb_end, irr_t)

        '''
        if len(t_megsa_i) > 0:
            if first_megsa:
                irr_megsa = irr_megsa_i
                t_megsa = t_megsa_i
                wl_megsa = wl_megsa
                first_megsa = False
            else:
                irr_megsa = np.concatenate([irr_megsa, irr_megsa_i], axis=0)
                t_megsa = np.concatenate([t_megsa, t_megsa_i], axis=0)
        if len(t_megsb_i) > 0:
            if first_megsb:
                irr_megsb = irr_megsb_i
                t_megsb = t_megsb_i
                wl_megsb = wl_megsb
                first_megsb = False
            else:
                irr_megsb = np.concatenate([irr_megsb, irr_megsb_i], axis=0)
                t_megsb = np.concatenate([t_megsb, t_megsb_i], axis=0)
        '''

        # Identify overlapping time
        t_megsab_i = np.intersect1d(t_megsa_i, t_megsb_i)
        if len(t_megsab_i) > 0:
            irr_megsab_i = (
                np.concatenate([np.stack([irr_megsa_i[np.where(t_megsa_i == t)[0]].flatten() for t in t_megsab_i],
                                         axis=0),
                                np.stack([irr_megsb_i[np.where(t_megsb_i == t)[0]].flatten() for t in t_megsab_i],
                                         axis=0)], axis=1))
            if first_megsab:
                irr_megsab = irr_megsab_i
                t_megsab = t_megsab_i
                wl_megsab = np.concatenate([wl_megsa, wl_megsb], axis=0)
                first_megsab = False
            else:
                irr_megsab = np.concatenate([irr_megsab, irr_megsab_i], axis=0)
                t_megsab = np.concatenate([t_megsab, t_megsab_i], axis=0)
        irr_megsa_i = None
        irr_megsb_i = None
        t_megsa_i = None
        t_megsb_i = None

    # Save netcdf
    # megs_a_path = os.path.join(savepath, f'{data_type}_MEGS-A.nc')
    # megs_b_path = os.path.join(savepath, f'{data_type}_MEGS-B.nc')
    # megs_ab_path = os.path.join(savepath, f'{data_type}_MEGS-AB.nc')
    # Create netcdf file
    # create_eve_netcdf(megs_a_path, irr_megsa, t_megsa, wl_megsa)
    # create_eve_netcdf(megs_b_path, irr_megsb, t_megsb, wl_megsb)
    create_eve_netcdf(savepath, irr_megsab, t_megsab, wl_megsab)
