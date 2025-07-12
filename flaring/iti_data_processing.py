from itipy.data.dataset import BaseDataset
from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, MapToDataEditor, AIAPrepEditor
import os
import glob
from astropy.io import fits
from astropy.io.fits import Header, PrimaryHDU
import tqdm as tqdm
import multiprocessing as mp
from functools import partial

# Configuration for all wavelengths to process
wavelengths = [94, 131, 171, 193, 211, 304]
base_input_folder = '/mnt/data/SDO-AIA'
output_folder = '/mnt/data2/AIA_processed_data'
os.makedirs(output_folder, exist_ok=True)


def process_file(fits_file, wavelength, resolution=512):
    """Process a single FITS file with the specified wavelength and resolution."""
    try:
        editors = [
            LoadMapEditor(),
            NormalizeRadiusEditor(resolution),
            AIAPrepEditor(calibration='auto'),
            MapToDataEditor()
        ]

        dataset = BaseDataset([fits_file], editors=editors, ext='.fits',
                              wavelength=wavelength, resolution=resolution)

        data, meta = dataset.convertData([fits_file])
        meta_header = meta['header']
        del meta_header['keycomments']

        # Create wavelength subfolder
        wavelength_folder = os.path.join(output_folder, str(wavelength))
        os.makedirs(wavelength_folder, exist_ok=True)

        output_file = os.path.join(wavelength_folder, os.path.basename(fits_file))
        fits.writeto(output_file, data, header=Header(meta_header), overwrite=True)
        return output_file
    except Exception as e:
        pass


def process_wavelength(wavelength):
    """Process files for a specific wavelength."""
    input_folder = os.path.join(base_input_folder, str(wavelength))

    # 🔎 Collect all .fits files
    fits_files = glob.glob(os.path.join(input_folder, '*.fits'))

    files_to_process = []
    skipped_count = 0

    for fits_file in fits_files:
        # Generate expected output filename (adjust this logic based on your process_file function)
        base_name = os.path.splitext(os.path.basename(fits_file))[0]
        output_file = os.path.join(output_folder, str(wavelength), f"{base_name}.fits")

        # Check if output file already exists
        if os.path.exists(output_file):
            skipped_count += 1
        else:
            files_to_process.append(fits_file)

    print(f"Found {len(files_to_process)} files for wavelength {wavelength}")
    print(f"Skipping {skipped_count} already processed files")
    print(f"Processing {len(files_to_process)} remaining files...")

    if not files_to_process:
        print("All files already processed!")
        return []

    print(f"Processing {len(fits_files)} files for wavelength {wavelength}...")

    # Create partial function with wavelength parameter fixed
    process_func = partial(process_file, wavelength=wavelength)

    # Process files with multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_func, files_to_process),
            total=len(files_to_process),
            desc=f"Processing {wavelength}Å files"
        ))

    return results


# Process all wavelengths
for wavelength in wavelengths:
    process_wavelength(wavelength)