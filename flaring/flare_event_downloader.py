from sunpy.net import Fido
from sunpy.net import attrs as a

class FlareEventDownloader:
    def __init__(self, start_date, end_date, event_type="FL", GOESCls="M1.0", directory=None):
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
        self.directory = directory

    def download_events(self):
        """
        Download flare events from the HEK database within the specified date range and filters.
        Returns a DataFrame containing the event data.
        """
        # Search for flare events in the specified date range
        result = Fido.search(
            a.Time(self.start_date, self.end_date),
            a.hek.EventType("FL"),
            a.hek.FL.GOESCls >= self.GOESCls,
            a.hek.OBS.Observatory == "GOES"
        )

        # Check if any results were found
        if len(result) == 0:
            print("No flare events found in the specified date range.")
            return None

        # Filter results to keep only relevant columns
        hek_results = result["hek"]
        print("Available columns:", hek_results.colnames)
        
        # Map to correct column names - HEK uses different naming conventions
        # Common HEK column names for flare events
        required_columns = []
        column_mapping = {
            'event_starttime': ['event_starttime', 'event_start_time', 'start_time'],
            'event_peaktime': ['event_peaktime', 'event_peak_time', 'peak_time'],
            'event_endtime': ['event_endtime', 'event_end_time', 'end_time'],
            'fl_goescls': ['fl_goescls', 'fl_goes_class', 'goes_class'],
            'ar_noaanum': ['ar_noaanum', 'ar_noaa_num', 'noaa_ar'],
            'hgc_coord': ['hgc_coord', 'hgc_longitude', 'hgc_latitude']
        }
        
        # Find which columns actually exist
        available_columns = []
        for desired_col, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in hek_results.colnames:
                    available_columns.append(possible_name)
                    break
            else:
                # If none of the possible names are found, check if any similar column exists
                similar_cols = [col for col in hek_results.colnames if any(part in col.lower() for part in desired_col.split('_'))]
                if similar_cols:
                    print(f"Warning: Could not find exact match for '{desired_col}', similar columns: {similar_cols}")
                    # Use the first similar column found
                    available_columns.append(similar_cols[0])
                else:
                    print(f"Warning: No column found for '{desired_col}'")
        
        # If no columns were found, return all available columns
        if not available_columns:
            print("No matching columns found, returning all available data")
            filtered_results = hek_results
        else:
            # Filter to only the columns we found
            filtered_results = hek_results[available_columns]

        # Convert to pandas DataFrame
        flare_df = filtered_results.to_pandas()
        
        # Ensure directory exists
        if self.directory:
            import os
            os.makedirs(self.directory, exist_ok=True)
            flare_df.to_csv(f"{self.directory}/flare_events_{self.start_date}_{self.end_date}.csv")
        
        return flare_df