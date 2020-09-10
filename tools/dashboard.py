from .library import get_output_path_from_s3_url
import pandas as pd
import urllib.request, json


class RideHailReference:
    def __init__(self, path_to_reference):
        self.ref_df = RideHailReference.taxi_usage_json_to_dataframes(path_to_reference)

    @staticmethod
    def __filter_by_year_month__(df, year_month):
        if year_month is not None:
            return df[df['year_month'] == year_month]
        else:
            return df

    def ref_get_trips_per_day(self, year_month=None):
        return RideHailReference.__filter_by_year_month__(
            self.ref_get_data_frame('trips_per_day', 'fhv_high_volume'), year_month)

    def ref_get_vehicles_per_day(self, year_month=None):
        return RideHailReference.__filter_by_year_month__(
            self.ref_get_data_frame('vehicles_per_day', 'fhv_high_volume'), year_month)

    def ref_get_trips_per_day_shared(self, year_month=None):
        return RideHailReference.__filter_by_year_month__(
            self.ref_get_data_frame('trips_per_day_shared', 'fhv_high_volume'), year_month)

    def get_reference(self, from_year_month):
        temp_df = self.ref_get_trips_per_day()
        trips_per_day_df = temp_df[(temp_df['year_month'] >= from_year_month)]

        temp_df = self.ref_get_vehicles_per_day()
        vehicles_per_day_df = temp_df[(temp_df['year_month'] >= from_year_month)]

        temp_df = self.ref_get_trips_per_day_shared()
        trips_per_day_shared_df = temp_df[(temp_df['year_month'] >= from_year_month)]
        result_df = pd.merge(trips_per_day_shared_df, pd.merge(trips_per_day_df, vehicles_per_day_df, on='year_month'),
                          on='year_month')
        result_df['trips_per_day_shared'].fillna(0, inplace=True)
        result_df = result_df[['year_month', 'trips_per_day', 'vehicles_per_day', 'trips_per_day_shared']].set_index('year_month')
        return result_df

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
            service_df['year_month'] = service_df['date'].dt.to_period('M')
            result[service] = service_df
        return result

    def ref_get_data_frame(self, column_name, taxi_service):
        return self.ref_df[taxi_service].reset_index()[[column_name, 'year_month']]

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

    def get_df(self):
        data = {'total_number_of_trips': [self.get_total_number_of_trips()], 'number_of_shared_trips': [self.get_number_of_shared_trips()]}
        return pd.DataFrame.from_dict(data)

    @staticmethod
    def get_scenarios_df(s3_baseline, others_dict):
        baseline = RideHailDashboard(s3_baseline, 10).get_df()
        baseline['scenario'] = 'baseline'
        baseline.set_index('scenario', inplace=True)
        df_list = [baseline]
        for key, value in others_dict.items():
            df = RideHailDashboard(value, 10).get_df()
            df['scenario'] = key
            df.set_index('scenario', inplace=True)
            df_list.append(df)
        return pd.concat(df_list)