import argparse
import logging
from datetime import datetime, timedelta
from download import download_sdo as sdo
import flare_event_downloader as fed

class FlareDownloadProcessor:
    def __init__(self, FlareEventDownloader, SDODownloader):
        """
        Initialize the FlareDownloadProcessor.
        This class is responsible for processing AIA flare downloads.
        """
        self.FlareEventDownloader = FlareEventDownloader
        self.SDODownloader = SDODownloader

    def process_download(self, time_before_start=timedelta(minutes=60), time_after_end=timedelta(minutes=0)):
        fl_events = self.FlareEventDownloader.download_events()
        for events in fl_events.iterrows():
            event = events[1]
            start_time = event['event_starttime'] - time_before_start
            end_time = event['event_endtime'] + time_after_end
            print(event)



            # Download SDO data for the flare event







if __name__ == '__main__':
    flare_event = fed.FlareEventDownloader("2023-06-01", "2023-06-03", event_type="FL", GOESCls="M1.0")
    sdo_downloader = sdo.SDODownloader("/home/griffingoodwin/downloads/sample_AIA","ggoodwin5@gsu.edu")
    FlareDownloadProcessor(flare_event, sdo_downloader).process_download()
    # parser = argparse.ArgumentParser(description='Download SDO data from JSOC with quality check and fallback')
    # parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    # parser.add_argument('--email', type=str, help='registered email address for JSOC.')
    # parser.add_argument('--start_date', type=str, help='start date in format YYYY-MM-DD.')
    # parser.add_argument('--end_date', type=str, help='end date in format YYYY-MM-DD.', required=False,
    #                     default=str(datetime.now()).split(' ')[0])
    # parser.add_argument('--cadence', type=int, help='cadence in minutes.', required=False, default=60)
    #
    # args = parser.parse_args()
    # download_dir = args.download_dir
    # start_date = args.start_date
    # end_date = args.end_date
    # cadence = args.cadence
