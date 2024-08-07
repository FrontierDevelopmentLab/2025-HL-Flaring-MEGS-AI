import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch
from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor
from sunpy.visualization.colormaps import cm
from irradiance.data.reprojection import transform
from skimage.measure import block_reduce
from sunpy.map import Map

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


def loadAIAMap(file_path, resolution=1024, map_reproject=False, calibration='auto', normalize_radius=False):
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

    try:
        s_map = AIAPrepEditor(calibration=calibration).call(s_map)
    except:
        s_map = AIAPrepEditor(calibration='aiapy').call(s_map)

    if normalize_radius:
        s_map = NormalizeRadiusEditor(resolution, padding_factor=0.1).call(s_map)
    else:
        downscale_factor = int(s_map.data.shape[0]/resolution)
        dummy_map = s_map.resample((resolution, resolution) * u.pix)
        binned_data = block_reduce(s_map.data, (downscale_factor,downscale_factor), func=np.mean)
        s_map = Map(binned_data, dummy_map.meta)

    if map_reproject:
        s_map = transform(s_map, lat=s_map.heliographic_latitude,
                          lon=s_map.heliographic_longitude, distance=1 * u.AU)
    return s_map


def loadMap(file_path, resolution=1024, map_reproject=False, calibration=None, normalize_radius=False):
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


def loadMapStack(file_paths:list, resolution:int=1024, remove_nans:bool=True, map_reproject:bool=False, aia_preprocessing:bool=True,
                 calibration:str='auto', apply_norm:bool=True, percentile_clip:float=None, normalize_radius=False)->np.array:
    """Load a stack of FITS files, resample ot specific resolution, and stack hem.

    Parameters
    ----------
    file_paths : list
        list of files to stack.
    resolution : int, optional
        target resolution, by default 1024
    remove_nans : bool, optional
        remove nans and infs, replace by 0, by default True
    map_reproject : bool, optional
        If to reproject the stack, by default False
    aia_preprocessing : bool, optional
        If to use aia preprocessing, by default True
    calibration : str, optional
        What type of AIA degradation fix to use, by default 'auto'
    apply_norm : bool, optional
        Whether to apply normalization after loading stack, by default True
    percentile_clip : float, optional
        If to apply a percentile clip. Number in percent: i.e. 0.25 means 0.25% NOT 25%, by default None

    Returns
    -------
    np.array
        returns AIA stack
    """    


    load_func = loadAIAMap if aia_preprocessing else loadMap
    s_maps = [load_func(file, resolution=resolution, map_reproject=map_reproject,
                        calibration=calibration, normalize_radius=normalize_radius) for file in file_paths]
    
    if apply_norm:
        stack = np.stack([sdo_norms[s_map.wavelength.value](s_map.data) for s_map in s_maps]).astype(np.float32)
    else:
        stack = np.stack([s_map.data for s_map in s_maps]).astype(np.float32)

    if remove_nans:
        stack[np.isnan(stack)] = 0
        stack[np.isinf(stack)] = 0

    if percentile_clip:
        for i in range(stack.shape[0]):
            percentiles = np.percentile(
                stack[i, :, :].reshape(-1), [100 - percentile_clip]
            )
            stack[i, :, :][stack[i, :, :] < 0] = 0
            stack[i, :, :][stack[i, :, :] > percentiles[0]] = percentiles[0]        

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
