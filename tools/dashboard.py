from .library import get_output_path_from_s3_url
import pandas as pd


class RideHailDashboard:
    def __init__(self, s3url, iteration):
        # General code to get output path fro S3 URL
        s3path = get_output_path_from_s3_url(s3url)
        self.passenger_per_trip_df = pd.read_csv(
            s3path + "/ITERS/it.{0}/{0}.passengerPerTripRideHail.csv".format(iteration))
        self.fleet_size = len(
            pd.read_csv(s3path + "/ITERS/it.{0}/{0}.rideHailFleet.csv.gz".format(iteration))['id'].unique())

    def get_number_of_shared_trips(self):
        return int(self.passenger_per_trip_df[["2", "3", "4", "5", "6"]].sum().sum())

    def get_total_number_of_trips(self):
        return int(self.passenger_per_trip_df[["1", "2", "3", "4", "5", "6"]].sum().sum())

    def get_fleet_size(self):
        return self.fleet_size
