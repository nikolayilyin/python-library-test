from library import get_output_path_from_s3_url
import pandas as pd


def load_events(events_path, chunk_filter, chunksize=100000):
    start_time = time.time()
    # Read first 20 rows in order to get all columns
    columns = pd.read_csv(events_path, low_memory=False, nrows=20).columns
    schema = {}
    # Make all of them `str` at first
    for col in columns:
        schema[col] = str
    # Assign correct type for specific columns
    schema["time"] = int
    schema["length"] = float
    schema['departureTime'] = pd.Int64Dtype()
    schema['arrivalTime'] = pd.Int64Dtype()

    df = pd.concat(
        [df[chunk_filter(df)] for df in pd.read_csv(events_path, low_memory=False, chunksize=chunksize, dtype=schema)])
    df['hour'] = (df['time'] / 3600).astype(int)
    print("events file url:", events_path)
    print("loading took %s seconds" % (time.time() - start_time))
    return df


def load_events_from_s3_chunked(s3url, iteration, chunk_filter, chunksize=100000):
    s3path = get_output_path_from_s3_url(s3url)
    events_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)
    return load_events(events_path, chunk_filter, chunksize)


def get_events_for_type(arg, event_type):
    df = None
    if isinstance(arg, pd.DataFrame):
        df = arg[arg['type'] == event_type]
    elif isinstance(arg, tuple):
        s3url = arg[0]
        iteration = arg[1]
        df = load_events_from_s3_chunked(s3url, iteration, lambda df: df['type'] == event_type)
    else:
        raise TypeError("Expect DataFrame or path, but got " + str(type(arg)))
    return df


def get_mode_choice(arg):
    def select_columns(df):
        return df[['time', 'hour', 'type', 'mode', 'person', 'currentTourMode', 'availableAlternatives',
                   'personalVehicleAvailable', 'expectedMaximumUtility', 'length', 'tourIndex', 'location']]

    df = get_events_for_type(arg, 'ModeChoice')
    df['hour'] = df['time'] // 3600
    return select_columns(df)


def get_replanning(arg):
    def select_columns(df):
        return df[['time', 'hour', 'type', 'reason', 'person']]

    df = get_events_for_type(arg, 'Replanning')
    df['hour'] = df['time'] // 3600
    return select_columns(df)


def get_path_traversal(arg):
    def select_columns(df):
        return df[
            ['time', 'hour', 'type', "length", "primaryFuelType", "secondaryFuelType", "primaryFuel", "secondaryFuel",
             "numPassengers", "links", "linkTravelTime", "mode", "departureTime",
             "arrivalTime", "vehicle", "driver", "vehicleType", "capacity", "startX", "startY", "endX", "endY",
             "primaryFuelLevel", "secondaryFuelLevel", "tollPaid", "seatingCapacity", "fromStopIndex", "toStopIndex"]]

    df = get_events_for_type(arg, 'PathTraversal')
    df = df.astype({"person": str, "departureTime": int, "arrivalTime": int})
    df['hour'] = df['time'] // 3600
    return select_columns(df)
