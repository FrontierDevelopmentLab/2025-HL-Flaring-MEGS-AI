import argparse
import logging
import os
from datetime import datetime, timedelta
from os import cpu_count

from download import download_sdo as sdo
from download import flare_event_downloader as fed
from download import sxr_downloader as sxr

class FlareDownloadProcessor:
    def __init__(self, FlareEventDownloader, SDODownloader, SXRDownloader, flaring_data=True):
        """
        Initialize the FlareDownloadProcessor.
        This class is responsible for processing AIA flare downloads.
        """
        self.FlareEventDownloader = FlareEventDownloader
        self.SDODownloader = SDODownloader
        self.SXRDownloader = SXRDownloader
        self.flaring_data = flaring_data

    def process_download(self, time_before_start=timedelta(minutes=15), time_after_end=timedelta(minutes=0)):

        fl_events = self.FlareEventDownloader.download_events()
        print(fl_events)
        [os.makedirs(os.path.join(self.SDODownloader.ds_path, str(c)), exist_ok=True) for c in
         [94, 131, 171, 193, 211, 304]]

        if self.flaring_data == True:
            print("Processing flare events...")
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
        elif self.flaring_data == False:
            print("Processing non-flare events...")
            start_time_fl = fl_events['event_starttime']
            end_time_fl = fl_events['event_endtime']
            for i, events in enumerate(fl_events.iterrows()):
                start_time = end_time_fl[i]
                end_time = start_time_fl[i+1] if i + 1 < len(fl_events) else end_time_fl[-1] + timedelta(minutes=5)
                self.SXRDownloader.download_and_save_goes_data(start_time.strftime('%Y-%m-%d'),
                                                               end_time.strftime('%Y-%m-%d'), max_workers= os.cpu_count())
                processed_dates = set()
                for d in [start_time + i * timedelta(minutes=1) for i in
                          range((end_time - start_time) // timedelta(minutes=1))]:
                    # Only download if we haven't processed this date yet
                    if d.isoformat() not in processed_dates:
                        self.SDODownloader.downloadDate(d)
                        processed_dates.add(d.isoformat())
                logging.info(f"Processed non-flare event {i + 1}/{len(fl_events)}: {start_time} to {end_time} ({len(processed_dates)})")
        else:
            raise ValueError("Invalid value for flaring_data. It should be either True or False.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download flare events and associated SDO data.')
    parser.add_argument('--start_date', type=str, default='2014-06-01',
                        help='Start date for downloading flare events (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2019-01-01',
                        help='End date for downloading flare events (YYYY-MM-DD)')
    parser.add_argument('--chunk_size', type=int, default=180,
                        help='Number of days per chunk for processing (default: 180)')
    parser.add_argument('--download_dir', type=str, default='/mnt/data',
                        help='Directory to save downloaded data (default: /mnt/data)')
    parser.add_argument('--flaring_data', dest='flaring_data', action='store_true',
                        help='Download flaring data (default)')
    parser.add_argument('--non_flaring_data', dest='flaring_data', action='store_false',
                        help='Download non-flaring data')
    parser.set_defaults(flaring_data=True)
    args = parser.parse_args()

    download_dir = args.download_dir
    start_date = args.start_date
    end_date = args.end_date
    chunk_size = args.chunk_size
    flaring_data = args.flaring_data
    # Parse start and end dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Process in chunks
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_size), end)
        print(f"Processing chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")

        sxr_downloader = sxr.SXRDownloader(f"{download_dir}/GOES-non-flaring",
                                           f"{download_dir}/GOES-non-flaring/combined")
        flare_event = fed.FlareEventDownloader(
            current_start.strftime("%Y-%m-%d"),
            current_end.strftime("%Y-%m-%d"),
            event_type="FL",
            GOESCls="M1.0",
            directory=f"{download_dir}/SDO-AIA-non-flaring/FlareEvents"
        )
        sdo_downloader = sdo.SDODownloader(f"{download_dir}/SDO-AIA-non-flaring", "ggoodwin5@gsu.edu")

        processor = FlareDownloadProcessor(flare_event, sdo_downloader, sxr_downloader,
                                           flaring_data=flaring_data)
        processor.process_download()

        current_start = current_end