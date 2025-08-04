import argparse
import logging
import os
import shutil
from datetime import timedelta, datetime

import pandas as pd
from astropy.io.fits import getheader
from sunpy.net import Fido, attrs as a
import sunpy_soar


class SOLODownloader:
    """
    Class to download Solar Orbiter data from the VSO.

    Args:
        base_path (str): Path to the directory where the downloaded data should be stored.
    """

    def __init__(self, base_path):
        self.base_path = base_path

        self.wavelengths_fsi = ['eui-fsi174-image', 'eui-fsi304-image']
        self.wavelengths_hri = ['eui-hrieuv174-image']
        self.dirs = ['eui-fsi174-image', 'eui-fsi304-image', 'eui-hrieuv174-image']
        [os.makedirs(os.path.join(base_path, dir), exist_ok=True) for dir in self.dirs]

    def downloadDate(self, date, FSI=True):
        """
        Download the data for the given date.

        Args:
            date (datetime): The date for which the data should be downloaded.
            FSI (bool): If True, download FSI data, else download HRI data.

        Returns:
            list: List of paths to the downloaded files.
        """
        files = []
        if FSI:
            try:
                # Download FSI
                for wl in self.wavelengths_fsi:
                    files += [self.downloadFSI(date, wl)]
                logging.info('Download complete %s' % date.isoformat())
            except Exception as ex:
                logging.error('Unable to download %s: %s' % (date.isoformat(), str(ex)))
                [os.remove(f) for f in files if os.path.exists(f)]
        else:
            try:
                # Download HRI
                for wl in self.wavelengths_hri:
                    files += [self.downloadHRI(date, wl)]
                logging.info('Download complete %s' % date.isoformat())
            except Exception as ex:
                # logging.error('Unable to download %s: %s' % (date.isoformat(), str(ex)))
                [os.remove(f) for f in files if os.path.exists(f)]

    def downloadFSI(self, query_date, wl):
        """
        Download the FSI data for the given date and wavelength.

        Args:
            query_date (datetime): The date for which the data should be downloaded.
            wl (str): The wavelength for which the data should be downloaded.

        Returns:
            str: Path to the downloaded file.
        """
        file_path = os.path.join(self.base_path, wl, "%s.fits" % query_date.isoformat("T", timespec='seconds'))
        if os.path.exists(file_path):
            return file_path
        #
        search = Fido.search(a.Time(query_date - timedelta(minutes=1), query_date + timedelta(minutes=1)),
                             a.Instrument('EUI'), a.soar.Product(wl), a.Level(2))
        assert search.file_num > 0, "No data found for %s (%s)" % (query_date.isoformat(), wl)
        search = sorted(search['soar'], key=lambda x: abs(pd.to_datetime(x['Start time']) - query_date).total_seconds())
        #
        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]

            # Clean data with header info or add printing meta data info
            header = getheader(file, 1)
            if header['CDELT1'] != 4.44012445:
                os.remove(file)
                continue

            # Check if the file is centered otherwise remove it
            naxis1 = header['NAXIS1']
            naxis2 = header['NAXIS2']
            euxcen = header['EUXCEN']
            euycen = header['EUYCEN']
            x_center = naxis1 / 2
            y_center = naxis2 / 2

            tolerance = 20 # pixels tolerance for center position
            if not (abs(euxcen - x_center) <= tolerance or abs(euycen - y_center) <= tolerance):
                print(f"Removing file {file} because it is not centered.")
                os.remove(file)
                continue

            # if header['SOOPTYPE'] not in ['LF1', 'LF2', 'LB1', 'RF1', 'RF2', 'RS5', 'RB1']:
            #    os.remove(file)
            #    continue

            shutil.move(file, file_path)
            return file_path

        raise Exception("No valid file found for %s (%s)!" % (query_date.isoformat(), wl))

    def downloadHRI(self, query_date, wl):
        """
        Download the HRI data for the given date and wavelength.

        Args:
            query_date (datetime): The date for which the data should be downloaded.
            wl (str): The wavelength for which the data should be downloaded.

        Returns:
            str: Path to the downloaded file.
        """
        file_path = os.path.join(self.base_path, wl, "%s.fits" % query_date.isoformat("T", timespec='seconds'))
        if os.path.exists(file_path):
            return file_path
        #
        search = Fido.search(a.Time(query_date - timedelta(minutes=3), query_date + timedelta(minutes=3)),
                             a.Instrument('EUI'), a.soar.Product(wl), a.Level(2))
        assert search.file_num > 0, "No data found for %s (%s)" % (query_date.isoformat(), wl)
        search = sorted(search['soar'], key=lambda x: abs(pd.to_datetime(x['Start time']) - query_date).total_seconds())
        #
        for entry in search:
            files = Fido.fetch(entry, path=self.base_path, progress=False)
            if len(files) != 1:
                continue
            file = files[0]
            # header = Map(file.meta)

            shutil.move(file, file_path)
            return file_path

        raise Exception("No valid file found for %s (%s)!" % (query_date.isoformat(), wl))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Solar Orbiter data')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=4)
    parser.add_argument('--start_date', type=str, help='start date in format YYYY-MM-DD.')
    parser.add_argument('--end_date', type=str, help='end date in format YYYY-MM-DD.', required=False,
                        default=str(datetime.now()).split(' ')[0])

    args = parser.parse_args()
    base_path = args.download_dir
    n_workers = args.n_workers
    start_date = args.start_date
    end_date = args.end_date

    start_date_datetime = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_date_datetime = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    download_util = SOLODownloader(base_path=base_path)

    for d in [start_date_datetime + i * timedelta(minutes=5) for i in
              range((end_date_datetime - start_date_datetime) // timedelta(minutes=5))]:
        download_util.downloadDate(d)