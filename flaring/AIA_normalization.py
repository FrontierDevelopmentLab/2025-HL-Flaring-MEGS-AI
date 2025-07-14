import sunpy.map
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import logging
import gc
from astropy.io import fits
from multiprocessing import Pool
import os

# Set up logging to capture debugging information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('aرفه_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_file(file):
    """
    Process a single FITS file: normalize by exposure time and compute mean intensity.
    Returns a tuple of (filename, result message, success flag).
    """
    try:
        logger.info(f"Processing {file.name}")

        # Load FITS file
        aia_map = sunpy.map.Map(file)

        # Check exposure time
        exptime = aia_map.exposure_time.value
        if exptime <= 0:
            logger.warning(f"{file.name}: Invalid exposure time ({exptime})")
            return file.name, f"Invalid exposure time ({exptime})", False

        # Normalize pixel data
        normalized_data = aia_map.data / exptime

        # Clean NaNs/Infs and negative values
        normalized_data = normalized_data[np.isfinite(normalized_data)]
        normalized_data = normalized_data[normalized_data > 0]

        if normalized_data.size == 0:
            logger.warning(f"{file.name}: No valid data after cleaning")
            return file.name, "No valid data after cleaning", False

        # Compute mean normalized intensity
        mean_val = np.mean(normalized_data)

        # Optionally save normalized FITS file
        output_dir = Path("/mnt/data/SDO-AIA/94_normalized")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / file.name
        hdu = fits.PrimaryHDU(normalized_data, header=aia_map.meta)
        hdu.writeto(output_file, overwrite=True)
        logger.info(f"{file.name}: Saved normalized data to {output_file}")

        # Free memory
        del aia_map, normalized_data
        gc.collect()

        return file.name, f"Mean normalized intensity = {mean_val:.2f}", True

    except Exception as e:
        logger.error(f"{file.name}: Failed - {str(e)}")
        return file.name, f"Failed - {str(e)}", False

def main():
    # Folder with AIA 94 Å files
    data_folder = Path("/mnt/data/SDO-AIA/94")
    fits_files = sorted(data_folder.glob("*.fits"))
    logger.info(f"Found {len(fits_files)} FITS files to process")

    # Optionally limit to a subset for debugging (e.g., around 27% mark)
    # start_idx = int(0.25 * len(fits_files))
    # end_idx = int(0.30 * len(fits_files))
    # fits_files = fits_files[start_idx:end_idx]

    # Use multiprocessing to speed up processing (adjust processes based on CPU cores)
    use_parallel = True  # Set to False for sequential processing
    if use_parallel:
        with Pool(processes=4) as pool:
            results = list(tqdm(
                pool.imap(process_file, fits_files),
                total=len(fits_files),
                desc="Normalizing AIA 94 wavelength by exposure time"
            ))
            # Log results
            for filename, result, success in results:
                if success:
                    logger.info(f"{filename}: {result}")
                else:
                    logger.warning(f"{filename}: {result}")
    else:
        # Sequential processing with detailed progress
        for i, file in enumerate(tqdm(fits_files, desc="Normalizing AIA 94 wavelength by exposure time")):
            logger.info(f"File {i+1}/{len(fits_files)}: {file.name}")
            filename, result, success = process_file(file)
            logger.info(f"{filename}: {result}")

if __name__ == "__main__":
    main()
