import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
from spi3s.data.eve_download import datetime_to_eve_format
from spi3s.data.eve_read import read_fits, read_evs, fits_to_df


def plot_spectra(wl, spectra, filename, fig_font=13, fig_format='png', fig_dpi=300, fig_transparent=False,
                 x_label='Wavelengths (Å)', y_label=r'Irradiance (W m$^{-2}$)', title_label='Spectra',
                 overplot=None):
    """Plot EVE spectra.

    Parameters
    ----------
    wl: array. Wavelengths of the spectra.
    spectra: array. Spectra to plot.
    fig_font: int. Font size of the figure.
    fig_format: str. Format of the figure.
    fig_dpi: int. DPI of the figure.

    Returns
    -------
    None
    """

    # Selection of plot colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Dimensions of the figure
    fig_lx = 4.0  # width of the figure in inches
    fig_ly = 4.0  # height of the figure in inches
    fig_lcb = 5  # colorbar width in percentage of the figure
    fig_left = 0.8  # left margin in inches
    fig_right = 0.4  # right margin in inches
    fig_bottom = 0.48  # bottom margin in inches
    fig_top = 0.32  # top margin in inches
    fig_wspace = 0.0  # width space between subplots in inches
    fig_hspace = 0.0  # height space between subplots in inches

    # Panel properties
    nrows = 1
    ncols = 1
    fig_sizex = 0. 
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols+1)).tolist() 
    fig_specy = np.zeros((nrows+1, ncols)).tolist()
    img_ratio = 1.0/2.25
    ncol_leg = 1
    for row in range(nrows):
        for col in range(ncols):
            if img_ratio > 1:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly*img_ratio + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx + fig_right
            else:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx/img_ratio + fig_right
            if row == 0 and col == ncols-1:
                fig_sizex = fig_specx[row][col+1]
            if row == nrows-1 and col == 0:
                fig_sizey = fig_specy[row+1][col]
    
    # Plot
    fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout = False)
    spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[0][0]/fig_sizex, 
                            right=-fig_right/fig_sizex + fig_specx[0][1]/fig_sizex,
                            bottom=fig_bottom/fig_sizey + fig_specy[0][0]/fig_sizey, 
                            top=-fig_top/fig_sizey + fig_specy[1][0]/fig_sizey,
                            wspace=0.00)
    ax = fig.add_subplot(spec[:, :])
    ax.grid(True, linewidth=0.25)
    ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=fig_font)
    ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=fig_font)
    ax.scatter(wl, spectra, s=2, color=colors[0], zorder=10)
    if overplot is not None:
        ax.plot(overplot[0], overplot[1], color='black', linestyle='--', linewidth=1.0, 
                zorder=10, label='MEGS-A/B cutoff')    
    ax.set_title(title_label, fontsize=fig_font+2, y=1.005)
    # Log scale
    ax.set_yscale('log')
    # Limit for y-axis
    ax.set_ylim([np.amin(spectra), 0.001])
    ax.set_xlim([np.floor(np.amin(wl))-10, np.ceil(np.amax(wl))+10])
    ax.set_ylabel(y_label, fontsize=fig_font, labelpad=5.0)
    ax.set_xlabel(x_label, fontsize=fig_font, labelpad=0.5)
    ax.get_xaxis().set_major_locator(plt.MaxNLocator(10))
    ax.legend(loc='best', numpoints=1, markerscale=4, fontsize=fig_font-1, labelspacing=0.2, ncol=ncol_leg)
    plt.draw()
    plt.savefig(filename, format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
    plt.close('all')


if "__main__" == __name__:
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
                        default='/home/benoit_tremblay_23/plots',
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

    # Read and plot data
    for i, evs in tqdm(enumerate(evs_files), total=len(evs_files)):
        # Read EVS file: Wavelengths, Time, Irradiance, and Binary Flags
        wl, irr_t, irr_data, bin_flags = read_evs(evs, verbose=False)

        # Extract MEGS-A and MEGS-B data as a function of time and wavelengths
        df_megsa_i = fits_to_df(irr_data, wl, wl_megsa_start, wl_megsa_end, irr_t, bin_flags)
        df_megsb_i = fits_to_df(irr_data, wl, wl_megsb_start, wl_megsb_end, irr_t, bin_flags)
        # Combined MEGS-A & MEGS-B Dataframe
        df_megsab_i = df_megsa_i.join(df_megsb_i, how='inner')
        nb_t, nb_wl = df_megsab_i.shape

        # Plot full spectrum
        for t in tqdm(range(nb_t)):
            # Extract date and irradiance values for each time step
            date = df_megsab_i.index[t]  # .strftime('%Y%m%dT%H%M%S')
            irr = df_megsab_i.iloc[t].values
            # Wavelengths - Convert to Å
            wavelengths = 10*df_megsab_i.columns.values
            # Title
            title = f"SDO/EVE Irradiance Spectrum - {date}"
            # Save
            filename = os.path.join(savepath, f"{data_type}_{date}_spectra.png")
            # Plot
            overplot = ((10*wl_megsa_end, 10*wl_megsa_end), (np.amin(irr), np.amax(irr)))
            plot_spectra(wavelengths, irr, filename, overplot=overplot, title_label=title)
            exit()
