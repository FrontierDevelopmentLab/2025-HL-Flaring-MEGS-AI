import numpy as np
import dateutil.parser as dt
from tqdm import tqdm
import argparse
from netCDF4 import Dataset
import os
import sys
import inspect

# Add 2024-HL-SPI3S/spi3s as a module
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))), '2024-HL-SPI3S')
sys.path.insert(0, parentdir)
from spi3s.data.eve_download import datetime_to_eve_format
from spi3s.data.eve_read import read_fits, read_evs, fits_to_df


def create_eve_netcdf(eve_raw_path, netcdf_outpath):
    """_summary_

    Args:
        eve_raw_path (str): path to directory of eve files
        netcdf_outpath (str): path to output cdf file

    Returns:
        netcdf file containing EVE data and metadata
    """
    os.makedirs(os.path.split(netcdf_outpath)[0], exist_ok=True)

    eve_date = np.load(eve_raw_path + "/iso.npy",allow_pickle=True)
    eve_irr = np.load(eve_raw_path + "/irradiance.npy",allow_pickle=True)
    eve_jd = np.load(eve_raw_path + "/jd.npy",allow_pickle=True)
    eve_logt = np.load(eve_raw_path + "/logt.npy",allow_pickle=True)
    eve_name = np.load(eve_raw_path + "/name.npy",allow_pickle=True)
    eve_wl = np.load(eve_raw_path + "/wavelength.npy",allow_pickle=True)

    ###################################################################################################################################################
    # CREATE NETCDF4 FILE
    ###################################################################################################################################################
    netcdfDB = Dataset(netcdf_outpath, "w", format="NETCDF4")
    netcdfDB.title = 'EVE spectral irradiance for specific spectral lines'

    # Create dimensions
    isoDate = netcdfDB.createDimension("isoDate", None)
    name = netcdfDB.createDimension("name", eve_name.shape[0])

    # Create variables and atributes
    isoDates = netcdfDB.createVariable('isoDate', 'S2', ('isoDate',))
    isoDates.units = 'string date in ISO format'

    julianDates = netcdfDB.createVariable('julianDate', 'f4', ('isoDate',))
    julianDates.units = 'days since the beginning of the Julian Period (January 1, 4713 BC)'

    names = netcdfDB.createVariable('name', 'S2', ('name',))
    names.units = 'strings with the line names'

    wavelength = netcdfDB.createVariable('wavelength', 'f4', ('name',))
    wavelength.units = 'line wavelength in nm'

    logt = netcdfDB.createVariable('logt', 'f4', ('name',))
    logt.units = 'log10 of the temperature'

    irradiance = netcdfDB.createVariable('irradiance', 'f4', ('isoDate','name',))
    irradiance.units = 'spectal irradiance in the specific line (w/m^2)'

    # Intialize variables
    isoDates[:] = eve_date
    julianDates[:] = eve_jd 
    names[:] = eve_name
    wavelength[:] = eve_wl
    logt[:] = eve_logt
    irradiance[:] = eve_irr

    netcdfDB.close()




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

    for i, evs in tqdm(enumerate(evs_files), total=len(evs_files)):
        # Read EVS file: Wavelengths, Time, Irradiance, and Binary Flags
        wl, irr_t, irr_data, bin_flags = read_evs(evs, verbose=False)
        # Dimensions
        nb_wl, nb_t = len(wl), len(irr_t)

        # Extract MEGS-A and MEGS-B data as a function of time and wavelengths
        df_megsa_i = fits_to_df(irr_data, wl, wl_megsa_start, wl_megsa_end, irr_t, bin_flags)
        df_megsb_i = fits_to_df(irr_data, wl, wl_megsb_start, wl_megsb_end, irr_t, bin_flags)
        # Combined MEGS-A & MEGS-B Dataframe
        df_megsab_i = df_megsa_i.join(df_megsb_i, how='inner')
        # Concatenate with previous dataframes
        if i == 0:
            df_megsa = df_megsa_i
            df_megsb = df_megsb_i
            df_megsab = df_megsab_i
        else:
            df_megsa = pd.concat([df_megsa, df_megsa_i], axis=0)
            df_megsb = pd.concat([df_megsb, df_megsb_i], axis=0)
            df_megsab = pd.concat([df_megsab, df_megsab_i], axis=0)
        print('MEGS-A dataframe: ', df_megsa.shape, ' / MEGS-B dataframe: ', df_megsb.shape, ' / MEGS-AB dataframe: ', df_megsab.shape)

    # Save dataframes
    nb_t_megsa, nb_t_megsb = df_megsa.shape[0], df_megsb.shape[0]
    if nb_t_megsa > 0:
        save_megsa = df_megsa.to_csv(os.path.join(savepath, f'{data_type}_MEGS-A.csv'))
    if nb_t_megsb > 0:
        save_megsb = df_megsb.to_csv(os.path.join(savepath, f'{data_type}_MEGS-B.csv'))
    if nb_t_megsa*nb_t_megsb > 0:
        save_megsab = df_megsab.to_csv(os.path.join(savepath, f'{data_type}_MEGS-AB.csv'))


    # Create netcdf file
    create_eve_netcdf(eve_path, output)