from sunpy.net import Fido
from sunpy.net import attrs as a

class FlareEventDownloader:
    def __init__(self, start_date, end_date, event_type="FL", GOESCls="M1.0"):
        """
        start_date: Start date for the flare event download.
        end_date: End date for the flare event download.
        event_type: Type of event to download, default is "FL" for flares.
        GOESCls: Minimum GOES class for filtering events, default is "M1.0".
        """
        self.start_date = start_date
        self.end_date = end_date
        self.event_type = event_type
        self.GOESCls = GOESCls

    def download_events(self):
        """
        Download flare events from the HEK database within the specified date range and filters.
        Returns a DataFrame containing the event data.
        """
        # Search for flare events in the specified date range
        result = Fido.search(
            a.Time(self.start_date, self.end_date),
            a.hek.EventType(self.event_type),
            a.hek.FL.GOESCls >= self.GOESCls,
            a.hek.OBS.Observatory == "GOES"
        )

        # Check if any results were found
        if len(result) == 0:
            print("No flare events found in the specified date range.")
            return None

        # Filter results to keep only relevant columns
        hek_results = result["hek"]
        filtered_results = hek_results["event_starttime", "event_peaktime", "event_endtime", "fl_goescls", "ar_noaanum"]

        # Convert to pandas DataFrame
        flare_df = filtered_results.to_pandas()

        return flare_df