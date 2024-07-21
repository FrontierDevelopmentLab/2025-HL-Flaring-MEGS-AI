from astropy.io import fits
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def read_fits(filename, hdu=None, verbose=False):
    """Read FITS file and return data and header.

    Parameters
    ----------
    filename: str. FITS file location.
    hdu: int. HDU to read from the FITS file.
    verbose: bool. Print FITS file info.

    Returns
    -------
    data: Data from the FITS file.
    header: Header from the FITS file.
    """
    with fits.open(filename) as hdul:
        # Print FITS file info
        if verbose:
            hdul.info()
        # If hdu is specified, return data, header, and column names
        return hdul[hdu].data, hdul[hdu].header, hdul[hdu].columns.names


def read_evs(filename, verbose=False):
    """Read EVE_V8 EVS FITS file and return time and irradiance.

    Parameters
    ----------
    filename: str. EVS FITS file location.
    verbose: bool. Print FITS file info.

    Returns
    -------
    wl: Wavelengths of spectra.
    time: Time of spectra.
    irr: Irradiance spectra.
    bin_flags: Binary flags to identify the quality of the data.
    """
    with fits.open(filename) as hdul:
        # Print FITS file info
        if verbose:
            hdul.info()
        # EVS - HDU 1 : Wavelength
        wl = hdul[1].data['WAVELENGTH']
        # EVS - HDU 3: Spectrum of Irradiance
        time = hdul[3].data['TAI']

        # TAI starts counting seconds from 1958-01-01
        epoch_1958 = datetime(1958, 1, 1)
        # Add the number of seconds to the epoch datetime
        time_iso = np.array([(epoch_1958 + timedelta(seconds=time[i])).isoformat() for i in range(len(time))])

        irr = hdul[3].data['IRRADIANCE']
        # flags = hdul[3].data['FLAGS']
        # sc_flags = hdul[3].data['SC_FLAGS']
        bin_flags = hdul[3].data['BIN_FLAGS']

        return wl, time_iso, irr, bin_flags


def fits_to_df(data, wl, wl_start, wl_end, t, flags):
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

    # Convert to dataframe
    df = pd.DataFrame(np.array(data_filtered).byteswap().newbyteorder(),
                      columns=np.array(wl_filtered).byteswap().newbyteorder(),
                      index=np.array(t_filtered).byteswap().newbyteorder())
    return df


def fits_to_arr(data, wl, wl_start, wl_end, t):
    """Convert FITS data to pandas dataframe.

    Parameters
    ----------
    data: Irradiance data from FITS file.
    wl: Wavelengths of spectra.
    wl_start: Start wavelength.
    wl_end: End wavelength.
    t: Time of spectra.

    Returns
    -------
    data_filtered: Irradiance data filtered in time and wavelengths. Rows = Time, Columns = Wavelengths.
    wl_filtered: Wavelengths of spectra.
    t_filtered: Time of spectra.
    """

    # Wavelength filtering
    wl_where = np.where((wl > wl_start) & (wl < wl_end))
    wl_filtered = wl[wl_where[0]]
    data_filtered = data[:, wl_where[0]]

    # Time filtering
    t_where = np.where(np.sum(data_filtered, axis=1) > 0)
    data_filtered = data_filtered[t_where[0]]
    t_filtered = t[t_where[0]]

    # Return filtered data
    return data_filtered, wl_filtered, t_filtered
