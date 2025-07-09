import argparse
import logging
import os
from datetime import datetime, timedelta
from os import cpu_count

from download import download_sdo as sdo
import flare_event_downloader as fed
import sxr_downloader as sxr

class FlareDownloadProcessor:
    def __init__(self, FlareEventDownloader, SDODownloader, SXRDownloader):
        """
        Initialize the FlareDownloadProcessor.
        This class is responsible for processing AIA flare downloads.
        """
        self.FlareEventDownloader = FlareEventDownloader
        self.SDODownloader = SDODownloader
        self.SXRDownloader = SXRDownloader

    def process_download(self, time_before_start=timedelta(minutes=60), time_after_end=timedelta(minutes=0)):

        fl_events = self.FlareEventDownloader.download_events()
        print(fl_events)
        [os.makedirs(os.path.join(self.SDODownloader.ds_path, str(c)), exist_ok=True) for c in
         [94, 131, 171, 193, 211, 304]]
        for i, events in enumerate(fl_events.iterrows()):
            event = events[1]
            start_time = event['event_starttime'] - time_before_start
            end_time = event['event_endtime'] + time_after_end
            self.SXRDownloader.download_and_save_goes_data(start_time.strftime('%Y-%m-%d'),
                                                           end_time.strftime('%Y-%m-%d'), max_workers= os.cpu_count())
            processed_dates = set()
            for d in [start_time + i * timedelta(minutes=1) for i in
                      range((end_time - start_time) // timedelta(minutes=1))]:
                # Only download if we haven't processed this date yet
                if d.isoformat() not in processed_dates:
                    self.SDODownloader.downloadDate(d)
                    processed_dates.add(d.isoformat())
            logging.info(f"Processed flare event {i + 1}/{len(fl_events)}: {event['event_starttime']} to {event['event_endtime']}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Download SDO data from JSOC with quality check and fallback for flare data')
    # parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    # parser.add_argument('--email', type=str, help='registered email address for JSOC.')
    # parser.add_argument('--start_date', type=str, help='start date in format YYYY-MM-DD.')
    # parser.add_argument('--end_date', type=str, help='end date in format YYYY-MM-DD.', required=False,
    #                      default=str(datetime.now()).split(' ')[0])
    # parser.add_argument('--time_before_start', type=int, help='', required=False)
    #
    # args = parser.parse_args()
    # download_dir = args.download_dir
    # start_date = args.start_date
    # end_date = args.end_date
    # cadence = args.cadence

    sxr_downloader = sxr.SXRDownloader("/mnt/data/GOES-additional", "/mnt/data/GOES-additional/combined")
    flare_event = fed.FlareEventDownloader("2012-01-01", "2025-01-01", event_type="FL", GOESCls="M1.0", directory="/mnt/data/SDO-AIA-additional/FlareEvents")
    sdo_downloader = sdo.SDODownloader("/mnt/data/SDO-AIA-additional","ggoodwin5@gsu.edu")
    FlareDownloadProcessor(flare_event, sdo_downloader,sxr_downloader).process_download()

