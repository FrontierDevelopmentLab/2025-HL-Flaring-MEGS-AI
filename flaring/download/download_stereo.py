import argparse
import logging
import os
import shutil
from datetime import timedelta, datetime
from glob import glob

import drms
import pandas as pd
from astropy import units as u
from sunpy.map import Map
from sunpy.net import Fido, attrs as a


class STEREODownloader:
    """
    Class to download STEREO SECCHI data from the VSO.

    Args:
        base_path (str): Path to the directory where the downloaded data should be stored.
    """

    def __init__(self, base_path):
        self.base_path = base_path
        self.wavelengths = [284, 195, 171]
        self.dirs = ['284', '195', '171']
        [os.makedirs(os.path.join(base_path, dir), exist_ok=True) for dir in self.dirs]

    def downloadDate(self, date):
        """
        Download the data for the given date.

        Args:
            sample (tuple): Tuple containing the date and source information.

        Returns:
            list: List of paths to the downloaded files.
        """

        files = []
        try:
            # Download SECCHI
            for wl in self.wavelengths:
                files += [self.downloadSECCHI(date, wl)]
            logging.info('Download complete %s' % date.isoformat())
        except Exception as ex:
            logging.error('Unable to download %s: %s' % (date.isoformat(), str(ex)))
            [os.remove(f) for f in files if os.path.exists(f)]

    def downloadSECCHI(self, query_date, wl):
        """
        Download the SECCHI data for the given date, wavelength and source.

        Args:
            query_date (datetime): The date for which the data should be downloaded.
            wl (int): The wavelength of the data.
            source (str): The source of the data.

        Returns:
            str: Path to the downloaded file.
        """
        file_path = os.path.join(self.base_path, str(wl), "%s.fits" % query_date.isoformat("T", timespec='seconds'))
        if os.path.exists(file_path):
            return file_path  # skip existing downloads (e.g. retry)
        #
        search = Fido.search(a.Time(query_date - timedelta(minutes=2), query_date + timedelta(minutes=2)),
                             a.Provider('SSC'), a.Instrument('SECCHI'), a.Wavelength(wl * u.AA))
        assert search.file_num > 0, "No data found for %s (%s)" % (query_date.isoformat(), wl)
        search = sorted(search['vso'], key=lambda x: abs(x['Start Time'].datetime.item() - query_date).total_seconds())
        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]
            try:
                header = Map(file).meta
            except:
                os.remove(file)
                continue
            if header['NAXIS1'] < 2048 or header['NAXIS2'] < 2048 or header['NMISSING'] != 0:
                os.remove(file)
                continue
            shutil.move(file, file_path)
            return file_path

        raise Exception("No valid file found for %s (%s)!" % (query_date.isoformat(), wl))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download STEREO data')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=4)
    parser.add_argument('--start_date', type=str, help='start date for the download.', required=False,
                        default='2008-05-01')
    parser.add_argument('--end_date', type=str, help='end date for the download.', required=False,
                        default=str(datetime.now()).split(' ')[0])
    parser.add_argument('--cadence', type=int, help='cadence for the download.', required=False, default=60)

    args = parser.parse_args()
    base_path = args.download_dir
    n_workers = args.n_workers
    start_date = args.start_date
    end_date = args.end_date
    cadence = args.cadence

    drms_client = drms.Client(email='robert.jarolim@uni-graz.at')
    download_util = STEREODownloader(base_path)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ])

    existing_dates = set(
        [os.path.basename(f).replace('.fits', '')[:10] for f in glob('/mnt/data2/STEREO-EUVI/**/*.fits')])

    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    # num_months = (end_date_datetime.year - start_date_datetime.year) * 12 + (end_date_datetime.month - start_date_datetime.month)
    # month_dates = [start_date_datetime + relativedelta(months=i) for i in range(num_months)]

    for d in [start_date_datetime + i * timedelta(minutes=3) for i in
              range((end_date_datetime - start_date_datetime) // timedelta(minutes=3))]:
        download_util.downloadDate(d)
