import numpy as np
from tqdm import tqdm
import argparse
from netCDF4 import Dataset
import os
import glob
from irradiance.data.download_eve import datetime_to_eve_format
from irradiance.data.read_eve import read_evs, fits_to_arr


def write_netcdf(filepath, irr, date, wl):
    """Write irradiance data to NetCDF file.

    Parameters
    ----------
    filepath: str. NetCDF file location.
    irr: Irradiance data.
    date: Date of irradiance data.
    wl: Wavelengths of spectra.

    Returns
    -------
    NetCDF file with irradiance data.

    Notes
    -----
    The NetCDF file will have the following structure:
    - Dimensions: Rows = Time, Columns = Wavelengths
    - Variables:
        - isoDate: Date in ISO format
        - wavelength: Wavelengths in nm
        - irradiance: Spectral irradiance in w/m^2
    """

    # Make sure the directory exists
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    # Create NetCDF file
    netcdfDB = Dataset(filepath, "w", format="NETCDF4")
    netcdfDB.title = 'EVE spectral irradiance'

    # Create dimensions: Rows = Time, Columns = Wavelengths
    isoDate = netcdfDB.createDimension("isoDate", None)
    wavelength = netcdfDB.createDimension("wavelength", wl.shape[0])

    # Create variables and attributes
    # Dates in Iso format
    isoDates = netcdfDB.createVariable('isoDate', 'S2', ('isoDate',))
    isoDates.units = 'Date in ISO format'
    # Wavelengths
    wavelengths = netcdfDB.createVariable('wavelength', 'f4', ('wavelength',))
    wavelengths.units = 'Wavelength in nm'
    # Irradiance
    irradiance = netcdfDB.createVariable('irradiance', 'f4', ('isoDate', 'wavelength',))
    irradiance.units = 'Spectral irradiance (w/m^2)'

    # Initialize variables
    isoDates[:] = date
    wavelengths[:] = wl
    irradiance[:] = irr

    # Save and close netcdf file
    netcdfDB.close()


def store_arr(irr_i, t_i, irr=None, t=None):
    """Store irradiance and time arrays.

    Parameters
    ----------
    irr: Irradiance array.
    irr_i: Irradiance array to store.
    t: Time array.
    t_i: Time array to store.

    Returns
    -------
    irr: Updated irradiance array.
    t: Updated time array.
    """

    if irr is None or len(t) == 0:
        return irr_i, t_i
    else:
        return np.concatenate([irr, irr_i], axis=0), np.concatenate([t, t_i], axis=0)


if __name__ == "__main__":
    """Read all EVE_V8 EVS FITS files and return store irradiance into dataframes.

        Parameters
        ----------
        start: str. Time to start downloading data in format YYYY-MM-DDTHH:MM:SS.
        end: str. Time to stop downloading data in format YYYY-MM-DDTHH:MM:SS.
        type: str. Data type: EVL (lines or bands) and EVS (spectra).
        instrument: str. Data instrument: MEGS-A, MEGS-B, MEGS-AB.
        level: str. Data level: 0, 1, 2, 2B, 3, 4.
        version: str. Data version. Currently only version 8 is available.
        data_dir: str. Local path to EVS data.
        save_dir: str. Local path to save data to.

        Returns
        -------
        cdf_megsa: NetCDF file with MEGS-A data.
        cdf_megsb: NetCDF file with MEGS-B data.
        cdf_megsab: NetCDF file with temporally overlapping MEGS-A and MEGS-B data.
        
        Notes:
        ------
        Level 2 spectra are the merged spectral measurements from the two spectrographs, MEGS-A and MEGS-B. The
        A detector is designed to measure from 6 –17 nm, and 17–37 nm using two filters, while the B detector is
        designed to measure 37–106. After the MEGS-A anomaly, MEGS-B was extended down to 33.33 nm.
        Source: https://lasp.colorado.edu/eve/data_access/eve_data/products/level2/EVE_L2_V8_README.pdf
    """
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-start', type=str, default='2010-01-01T00:00:00',
                        help='Enter start time for data downloading in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('-end', type=str, default='2015-01-01T00:00:00',
                        help='Enter end time for data downloading in format YYYY-MM-DDTHH:MM:SS')
    parser.add_argument('-type', type=str, default='EVS',
                        help='Specify data type: EVL (lines or bands) and EVS (spectra)')
    parser.add_argument('-instrument', type=str,  nargs="+", default=['MEGS-AB'],
                        help='Specify instrument: MEGS-A, MEGS-B, MEGS-AB')
    parser.add_argument('-level', type=str, default='2B',
                        help='Specify data level: 0, 1, 2, 2B, 3, 4')
    parser.add_argument('-version', type=str, default='008',
                        help='Specify data version')
    parser.add_argument('-revision', type=str, default='01',
                        help='Specify data revision number')
    parser.add_argument('-data_dir', type=str,
                        default='/mnt/disks/data-spi3s-irradiance/EVE',
                        help='Specify where data is stored locally')
    parser.add_argument('-save_dir', type=str,
                        default=None,
                        help='Specify where to store data using a path to a folder.  Creates automatic naming.')
    parser.add_argument('-save_path', type=str,
                        default=None,
                        help='Specify the file where to store the data, overrides any automatic generation of paths')                        
    args = parser.parse_args()

    # TODO: Modify for EVL dataset

    # Pass arguments to variables
    start_date = args.start
    end_date = args.end
    data_type = args.type
    data_instrument = args.instrument
    data_level = args.level
    data_version = args.version
    data_revision = args.revision
    data_dir = args.data_dir
    save_dir = args.save_dir
    save_path = args.save_path

    if save_dir is None and save_path is None:
        raise Exception("Either save_dir or save_path must be specified.")
    
    megs_output_path = None
    if save_path is not None:
        megs_output_path = save_path

    # Wavelengths - MEGS-A
    wl_megsa_start, wl_megsa_end = 5.77, 33.33
    # Wavelengths - MEGS-B
    wl_megsb_start, wl_megsb_end = wl_megsa_end, 106.61

    # Start & end times in EVE format
    start_year, start_yday, start_hour, start_min = datetime_to_eve_format(start_date)
    end_year, end_yday, end_hour, end_min = datetime_to_eve_format(end_date)

    # Files
    eve_files = sorted(glob.glob(os.path.join(data_dir,
                                              f"{data_type}_L{data_level}_*_{data_version}_{data_revision}.fit.gz")))

    # Check data type: EVL for spectral lines or EVS for full spectra
    if data_type == 'EVL':
        raise ValueError('This script is only for EVS data. Please specify EVS as data type.')

    elif data_type == 'EVS':

        # Extract files between start time and end time. Take advantage of YYYYDDD format for easy comparison.
        eve_files = \
            [e for e in eve_files if 1000*start_year+start_yday <= int(e.split('_')[2]) <= 1000*end_year+end_yday]

        # Initialize arrays
        irr_megsa, t_megsa = None, None
        irr_megsb, t_megsb = None, None
        irr_megsab, t_megsab = None, None

        # Loop over files
        for i, eve in tqdm(enumerate(eve_files), total=len(eve_files)):

            # Read EVS file: Wavelengths, Time, Irradiance, and Binary Flags
            wl, irr_t, irr_data, bin_flags = read_evs(eve)
            # Dimensions
            nb_wl, nb_t = len(wl), len(irr_t)

            # Extract MEGS-A and MEGS-B data as a function of time and wavelengths
            irr_megsa_i, wl_megsa, t_megsa_i = fits_to_arr(irr_data, wl, wl_megsa_start, wl_megsa_end, irr_t)
            irr_megsb_i, wl_megsb, t_megsb_i = fits_to_arr(irr_data, wl, wl_megsb_start, wl_megsb_end, irr_t)

            # Check if MEGS-A data is available and if user wants to save it
            if 'MEGS-A' in data_instrument and len(t_megsa_i) > 0:
                # Store MEGS-A data
                irr_megsa, t_megsa = store_arr(irr_megsa_i, t_megsa_i, irr=irr_megsa, t=t_megsa)

                # Create netcdf file
                if i == len(eve_files) - 1:
                    if megs_output_path is None:
                        megs_output_path = os.path.join(save_dir, f'{data_type}_MEGS-A_irradiance.nc')
                    write_netcdf(megs_output_path, irr_megsa, t_megsa, wl_megsa)

            # Check if MEGS-B data is available and if user wants to save it
            if 'MEGS-B' in data_instrument and len(t_megsb_i) > 0:
                # Store MEGS-B data
                irr_megsb, t_megsb = store_arr(irr_megsb_i, t_megsb_i, irr=irr_megsb, t=t_megsb)

                # Create netcdf file
                if i == len(eve_files) - 1:
                    if megs_output_path is None:
                        megs_output_path = os.path.join(save_dir, f'{data_type}_MEGS-B_irradiance.nc')
                    write_netcdf(megs_output_path, irr_megsb, t_megsb, wl_megsb)

            # Identify overlap between MEGS-A and MEGS-B
            t_megsab_i = np.intersect1d(t_megsa_i, t_megsb_i)
            if 'MEGS-AB' in data_instrument and len(t_megsab_i) > 0:
                # Extract MEGS-AB data
                irr_megsab_i = (
                    np.concatenate([np.stack([irr_megsa_i[np.where(t_megsa_i == t)[0]].flatten() for t in t_megsab_i],
                                             axis=0),
                                    np.stack([irr_megsb_i[np.where(t_megsb_i == t)[0]].flatten() for t in t_megsab_i],
                                             axis=0)], axis=1))
                # Store MEGS-AB data
                irr_megsab, t_megsab = store_arr(irr_megsab_i, t_megsab_i, irr=irr_megsab, t=t_megsab)

                # Create netcdf file
                if i == len(eve_files) - 1:
                    if megs_output_path is None:
                        megs_output_path = os.path.join(save_dir, f'{data_type}_MEGS-AB_irradiance.nc')
                    write_netcdf(megs_output_path, irr_megsab, t_megsab, np.concatenate([wl_megsa, wl_megsb]))

            # Release memory
            irr_megsa_i = None
            irr_megsb_i = None
            irr_megsab_i = None
            t_megsa_i = None
            t_megsb_i = None

    else:
        raise ValueError('Data type must be EVS or EVL.')
