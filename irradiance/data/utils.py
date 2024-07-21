import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch
from iti.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor
from sunpy.visualization.colormaps import cm
from irradiance.data.reprojection import transform

sdo_img_norm = ImageNormalize(vmin=0, vmax=1, stretch=LinearStretch(), clip=True)

psi_norms = {171: ImageNormalize(vmin=0, vmax=22348.267578125, stretch=AsinhStretch(0.005), clip=True),
             193: ImageNormalize(vmin=0, vmax=50000, stretch=AsinhStretch(0.005), clip=True),
             211: ImageNormalize(vmin=0, vmax=13503.1240234375, stretch=AsinhStretch(0.005), clip=True), }

sdo_cmaps = {171: cm.sdoaia171, 193: cm.sdoaia193, 211: cm.sdoaia211, 304: cm.sdoaia304}

sdo_norms = {94: ImageNormalize(vmin=0, vmax=340, clip=True),
             131: ImageNormalize(vmin=0, vmax=1400, clip=True),
             171: ImageNormalize(vmin=0, vmax=8600, clip=True),
             193: ImageNormalize(vmin=0, vmax=9800, clip=True),
             211: ImageNormalize(vmin=0, vmax=5800, clip=True),
             304: ImageNormalize(vmin=0, vmax=8800, clip=True),
             335: ImageNormalize(vmin=0, vmax=600, clip=True),
             1600: ImageNormalize(vmin=0, vmax=4000, clip=True),
             1700: ImageNormalize(vmin=0, vmax=4000, clip=True)
             }


def loadAIAMap(file_path, resolution=1024, map_reproject=False, calibration='auto'):
    """Load and preprocess AIA file to make them compatible to ITI.


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.
    map_reproject: apply preprocessing to remove off-limb (map to heliographic map and transform back to original view).
    calibration: calibration mode for AIAPrepEditor

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    assert s_map.meta['QUALITY'] == 0, f'Invalid quality flag while loading AIA Map: {s_map.meta["QUALITY"]}'
    s_map = NormalizeRadiusEditor(resolution, padding_factor=0.1).call(s_map)
    try:
        s_map = AIAPrepEditor(calibration=calibration).call(s_map)
    except:
        s_map = AIAPrepEditor(calibration='aiapy').call(s_map)
    if map_reproject:
        s_map = transform(s_map, lat=s_map.heliographic_latitude,
                          lon=s_map.heliographic_longitude, distance=1 * u.AU)
    return s_map


def loadMap(file_path, resolution=1024, map_reproject=False, calibration=None):
    """Load and resample a FITS file (no pre-processing).


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.
    map_reproject: apply preprocessing to remove off-limb (map to heliographic map and transform back to original view).
    calibration: calibration mode

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    s_map = s_map.resample((resolution, resolution) * u.pix)
    # s_map = NormalizeRadiusEditor(resolution, padding_factor=0.225).call(s_map)
    if map_reproject:
        s_map = transform(s_map, lat=s_map.heliographic_latitude,
                          lon=s_map.heliographic_longitude, distance=1 * u.AU)
    return s_map


def loadMapStack(file_paths, resolution=1024, remove_nans=True, map_reproject=False, aia_preprocessing=True,
                 calibration='auto'):
    """Load a stack of FITS files, resample ot specific resolution, and stack hem.


    Parameters
    ----------
    file_paths: list of files to stack.
    resolution: target resolution in pixels of 2.2 solar radii.

    Returns
    -------
    numpy array with AIA stack
    """
    load_func = loadAIAMap if aia_preprocessing else loadMap
    s_maps = [load_func(file, resolution=resolution, map_reproject=map_reproject,
                        calibration=calibration) for file in file_paths]
    # Clip extreme values and normalize in the process
    stack = np.stack([sdo_norms[s_map.wavelength.value](s_map.data) for s_map in s_maps]).astype(np.float32)

    if remove_nans:
        stack[np.isnan(stack)] = 0
        stack[np.isinf(stack)] = 0

    return stack.data


def str2bool(v):
    """converts string to boolean

        arguments
        ----------
        v: string
            string to convert to boolean

        Returns
        -------
        a boolean value based on the string placed
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
