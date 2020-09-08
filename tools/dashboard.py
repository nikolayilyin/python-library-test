from .library import get_output_path_from_s3_url
import pandas as pd
import urllib.request, json


class RideHailReference:
    def __init__(self, path_to_reference):
        self.ref_df = RideHailReference.taxi_usage_json_to_dataframes(path_to_reference)

    def ref_get_trips_per_day(self):
        return self.ref_get_data_frame('trips_per_day', ['fhv_high_volume'])

    def ref_get_vehicles_per_day(self):
        return self.ref_get_data_frame('vehicles_per_day', ['fhv_high_volume'])

    def ref_get_trips_per_day_shared(self):
        return self.ref_get_data_frame('trips_per_day_shared', ['fhv_high_volume'])

    @staticmethod
    def taxi_usage_json_to_dataframes(json_path):
        json_url = urllib.request.urlopen(json_path)
        data = json.loads(json_url.read())
        data.pop('tlc_date', None)
        data.pop('fhv_date', None)

        df = pd.read_json(json.dumps(data))
        taxi_services = ['fhv_black_car', 'fhv_high_volume', 'fhv_livery', 'fhv_lux_limo', 'green', 'yellow', 'juno',
                         'lyft', 'uber', 'via']
        result = {}
        for service in taxi_services:
            service_df = df[[service]].T.reset_index().apply(pd.Series.explode)
            service_df.sort_values(by=['month'], inplace=True)
            service_df['date'] = service_df['month'].transform(lambda x: pd.to_datetime(x, unit='ms'))
            service_df.set_index('date', inplace=True)
            result[service] = service_df
        return result

    def ref_get_data_frame(self, column_name, taxi_services=[]):
        result_df = {}
        if taxi_services == []:
            taxi_services = self.ref_df.keys()
        for service in taxi_services:
            df = self.ref_df[service]
            reseted_index = df.reset_index()
            result_df[service] = df[column_name]
        return pd.concat(result_df, join='outer', axis=1)


class RideHailDashboard:
    def __init__(self, path_to_reference, s3url, iteration):
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
