"""
this is compilation of useful functions that might be helpful to analyse BEAM-related data
"""
from urllib.error import HTTPError

import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
import urllib
import pandas as pd
import re

from urllib import request
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from io import StringIO
import statistics


# import dashboard.ridehail_dashboard
# import events.events
# import routing.routing

def get_output_path_from_s3_url(s3_url):
    """
    transform s3 output path (from beam runs spreadsheet) into path to s3 output
    that may be used as part of path to the file.

    s3path = get_output_path_from_s3_url(s3url)
    beam_log_path = s3path + '/beamLog.out'
    """
    return s3_url \
        .strip() \
        .replace("s3.us-east-2.amazonaws.com/beam-outputs/index.html#", "beam-outputs.s3.amazonaws.com/")


def get_realized_modes_as_str(full_path, data_file_name='referenceRealizedModeChoice.csv'):
    if data_file_name not in full_path:
        path = get_output_path_from_s3_url(full_path) + "/" + data_file_name
    else:
        path = get_output_path_from_s3_url(full_path)

    df = pd.read_csv(path,
                     names=['bike', 'car', 'cav', 'drive_transit', 'ride_hail', 'ride_hail_pooled', 'ride_hail_transit',
                            'walk', 'walk_transit'])
    last_row = df.tail(1)
    car = float(last_row['car'])
    walk = float(last_row['walk'])
    bike = float(last_row['bike'])
    ride_hail = float(last_row['ride_hail'])
    ride_hail_transit = float(last_row['ride_hail_transit'])
    walk_transit = float(last_row['walk_transit'])
    drive_transit = float(last_row['drive_transit'])
    ride_hail_pooled = float(last_row['ride_hail_pooled'])
    # car	walk	bike	ride_hail	ride_hail_transit	walk_transit	drive_transit	ride_hail_pooled
    result = "%f,%f,%f,%f,%f,%f,%f,%f" % (
        car, walk, bike, ride_hail, ride_hail_transit, walk_transit, drive_transit, ride_hail_pooled)
    return result


def plot_simulation_vs_google_speed_comparison(s3url, iteration, compare_vs_3am, title=""):
    s3path = get_output_path_from_s3_url(s3url)
    google_tt = pd.read_csv(s3path + "/ITERS/it.{0}/{0}.googleTravelTimeEstimation.csv".format(iteration))

    google_tt_3am = google_tt[google_tt['departureTime'] == 3 * 60 * 60].copy()
    google_tt_rest = google_tt[
        (google_tt['departureTime'] != 3 * 60 * 60) & (google_tt['departureTime'] < 24 * 60 * 60)].copy()

    google_tt_column = 'googleTravelTimeWithTraffic'
    google_tt_column3am = 'googleTravelTimeWithTraffic'

    def get_speed(distance, travel_time):
        # travel time may be -1 for some google requests because of some google errors
        if travel_time <= 0:
            return 0
        else:
            return distance / travel_time

    def get_uid(row):
        return "{}:{}:{}:{}:{}".format(row['vehicleId'], row['originLat'], row['originLng'], row['destLat'],
                                       row['destLng'])

    if compare_vs_3am:
        google_tt_3am['googleDistance3am'] = google_tt_3am['googleDistance']
        google_tt_3am['google_api_speed_3am'] = google_tt_3am.apply(
            lambda row: (get_speed(row['googleDistance'], row[google_tt_column3am])), axis=1)

        google_tt_3am['uid'] = google_tt_3am.apply(get_uid, axis=1)
        google_tt_3am = google_tt_3am.groupby('uid')['uid', 'google_api_speed_3am', 'googleDistance3am'] \
            .agg(['min', 'mean', 'max']).copy()
        google_tt_3am.reset_index(inplace=True)

    google_tt_rest['google_api_speed'] = google_tt_rest.apply(
        lambda row: (get_speed(row['googleDistance'], row[google_tt_column])), axis=1)
    google_tt_rest['sim_speed'] = google_tt_rest.apply(lambda row: (get_speed(row['legLength'], row['simTravelTime'])),
                                                       axis=1)
    google_tt_rest['uid'] = google_tt_rest.apply(get_uid, axis=1)

    df = google_tt_rest \
        .groupby(['uid', 'departureTime'])[[google_tt_column, 'googleDistance', 'google_api_speed', 'sim_speed']] \
        .agg({google_tt_column: ['min', 'mean', 'max'],
              'googleDistance': ['min', 'mean', 'max'],
              'google_api_speed': ['min', 'mean', 'max'], 'sim_speed': ['min']}) \
        .copy()

    df.reset_index(inplace=True)

    if compare_vs_3am:
        df = df.join(google_tt_3am.set_index('uid'), on='uid')

    df['departure_hour'] = df['departureTime'] // 3600

    df.columns = ['{}_{}'.format(x[0], x[1]) for x in df.columns]
    df['sim_speed'] = df['sim_speed_min']

    fig, (ax00, ax0, ax1) = plt.subplots(1, 3, figsize=(19, 3))
    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(wspace=0.15, hspace=0.1)
    fig.suptitle(title, y=1.11)

    title0 = "simulation speed - Google speed"
    title1 = "simulation speed comparison with Google speed"
    if compare_vs_3am:
        title0 = title0 + " at 3am"
        title1 = title1 + " at 3am"

    def plot_hist(google_column_name, label):
        result_name = 'error_' + label
        df[result_name] = df['sim_speed'] - df[google_column_name]
        bins = range(-19, 19, 2)
        # df[result_name].hist(bins=bins, alpha=1, histtype='step', linewidth=3, label=label, ax=ax0)
        df[result_name].hist(bins=bins, alpha=0.3, label=label, ax=ax0)
        df[result_name].plot.kde(bw_method=0.2, ax=ax00)

    if compare_vs_3am:
        plot_hist('google_api_speed_3am_max', 'max')
    else:
        plot_hist('google_api_speed_max', 'max')
        plot_hist('google_api_speed_mean', 'mean')
        plot_hist('google_api_speed_min', 'min')

    ax00.axvline(0, color="black", linestyle="--")
    ax00.set_title(title0)
    ax00.legend(loc='upper left')

    ax0.axvline(0, color="black", linestyle="--")
    ax0.set_title(title0)
    ax0.set_xlabel('Speed difference')
    ax0.set_ylabel('Frequency')
    ax0.legend(loc='upper left')

    to_plot_df_speed_0 = df.groupby(['departure_hour_']).mean()
    to_plot_df_speed_0['departure_hour_'] = to_plot_df_speed_0.index

    if compare_vs_3am:
        to_plot_df_speed_0.plot(x='departure_hour_', y='google_api_speed_3am_min', label='g 3am min', ax=ax1)
        to_plot_df_speed_0.plot(x='departure_hour_', y='google_api_speed_3am_mean', label='g 3am mean', ax=ax1)
        to_plot_df_speed_0.plot(x='departure_hour_', y='google_api_speed_3am_max', label='g 3am max', ax=ax1)
    else:
        to_plot_df_speed_0.plot(x='departure_hour_', y='google_api_speed_min', label='g min', ax=ax1)
        to_plot_df_speed_0.plot(x='departure_hour_', y='google_api_speed_mean', label='g mean', ax=ax1)
        to_plot_df_speed_0.plot(x='departure_hour_', y='google_api_speed_max', label='g max', ax=ax1)

    to_plot_df_speed_0.plot(x='departure_hour_', y='sim_speed', ax=ax1)

    ax1.legend()
    ax1.set_title(title1)


def print_network_from(s3path, take_rows):
    output = get_output_path_from_s3_url(s3path)
    path = output + '/network.csv.gz'
    network_df = show_network(path, take_rows)
    print(str(take_rows) + " max link types from network from run:     " + s3path.split('/')[-1])
    print(network_df)
    print("")


def show_network(path, take_rows=0):
    network_df = pd.read_csv(path)
    network_df = network_df[['attributeOrigType', 'linkId']]
    grouped_df = network_df.groupby(['attributeOrigType']).count()
    grouped_df.sort_values(by=['linkId'], inplace=True)
    if take_rows == 0:
        return grouped_df
    else:
        return grouped_df.tail(take_rows)


def print_file_from_url(file_url):
    file = urllib.request.urlopen(file_url)
    for b_line in file.readlines():
        print(b_line.decode("utf-8"))


def grep_beamlog(url, keywords):
    file = urllib.request.urlopen(url)
    for b_line in file.readlines():
        line = b_line.decode("utf-8")
        for keyword in keywords:
            if keyword in line:
                print(line)


def read_traffic_counts(df):
    df['date'] = df['Date'].apply(lambda x: dt.datetime.strptime(x, "%m/%d/%Y"))
    df['hour_0'] = df['12:00-1:00 AM']
    df['hour_1'] = df['1:00-2:00AM']
    df['hour_2'] = df['2:00-3:00AM']
    df['hour_3'] = df['2:00-3:00AM']
    df['hour_4'] = df['3:00-4:00AM']
    df['hour_5'] = df['4:00-5:00AM']
    df['hour_6'] = df['5:00-6:00AM']
    df['hour_7'] = df['6:00-7:00AM']
    df['hour_8'] = df['7:00-8:00AM']
    df['hour_9'] = df['9:00-10:00AM']
    df['hour_10'] = df['10:00-11:00AM']
    df['hour_11'] = df['11:00-12:00PM']
    df['hour_12'] = df['12:00-1:00PM']
    df['hour_13'] = df['1:00-2:00PM']
    df['hour_14'] = df['2:00-3:00PM']
    df['hour_15'] = df['3:00-4:00PM']
    df['hour_16'] = df['4:00-5:00PM']
    df['hour_17'] = df['5:00-6:00PM']
    df['hour_18'] = df['6:00-7:00PM']
    df['hour_19'] = df['7:00-8:00PM']
    df['hour_20'] = df['8:00-9:00PM']
    df['hour_21'] = df['9:00-10:00PM']
    df['hour_22'] = df['10:00-11:00PM']
    df['hour_23'] = df['11:00-12:00AM']
    df = df.drop(['Date', '12:00-1:00 AM', '1:00-2:00AM', '2:00-3:00AM', '3:00-4:00AM', '4:00-5:00AM', '5:00-6:00AM',
                  '6:00-7:00AM', '7:00-8:00AM', '8:00-9:00AM',
                  '9:00-10:00AM', '10:00-11:00AM', '11:00-12:00PM', '12:00-1:00PM', '1:00-2:00PM', '2:00-3:00PM',
                  '3:00-4:00PM', '4:00-5:00PM', '5:00-6:00PM',
                  '6:00-7:00PM', '7:00-8:00PM', '8:00-9:00PM', '9:00-10:00PM', '10:00-11:00PM', '11:00-12:00AM'],
                 axis=1)
    return df


def aggregate_per_hour(traffic_df, date):
    wednesday_df = traffic_df[traffic_df['date'] == date]
    agg_df = wednesday_df.groupby(['date']).sum()
    agg_list = []
    for i in range(0, 24):
        xs = [i, agg_df['hour_%d' % i][0]]
        agg_list.append(xs)
    return pd.DataFrame(agg_list, columns=['hour', 'count'])


def plot_traffic_count(date):
    # https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts-2014-2018-/ertz-hr4r
    path_to_csv = 'https://data.cityofnewyork.us/api/views/ertz-hr4r/rows.csv?accessType=DOWNLOAD'
    df = read_traffic_counts(pd.read_csv(path_to_csv))
    agg_per_hour_df = aggregate_per_hour(df, date)
    agg_per_hour_df.plot(x='hour', y='count', title='Date is %s' % date)


def get_calibration_png_graphs(s3url, first_iteration=0, last_iteration=0, png_title=None):
    s3path = get_output_path_from_s3_url(s3url)

    # ######
    # fig = plt.figure(figsize=(8, 6))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    # ax0 = plt.subplot(gs[0])
    # ax0.plot(x, y)
    # ax1 = plt.subplot(gs[1])
    # ax1.plot(y, x)
    # ######

    def display_two_png(path1, path2, title=png_title):
        def display_png(ax, path):
            ax_title = path.split('/')[-1] + "\n"

            ax.set_title(ax_title, pad=0.1)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_xaxis().labelpad = 0
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_yaxis().labelpad = 0
            ax.imshow(plt.imread(path))

        fig, axs = plt.subplots(1, 2, figsize=(25, 10))
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.tight_layout()

        display_png(axs[0], path1)
        display_png(axs[1], path2)
        plt.suptitle(title)

    display_two_png(s3path + "/stopwatch.png",
                    s3path + "/AverageCarSpeed.png")

    display_two_png(s3path + "/ITERS/it.{0}/{0}.AverageSpeed.Personal.png".format(first_iteration),
                    s3path + "/ITERS/it.{0}/{0}.AverageSpeed.Personal.png".format(last_iteration))

    display_two_png(s3path + "/referenceRealizedModeChoice.png",
                    s3path + "/referenceRealizedModeChoice_commute.png")


def analyze_vehicle_passenger_by_hour(s3url, iteration):
    s3path = get_output_path_from_s3_url(s3url)
    events_file_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)
    plot_vehicle_type_passengets_by_hours(events_file_path)


def plot_vehicle_type_passengets_by_hours(events_file_path, chunksize=100000):
    events = pd.concat([events[events['type'] == 'PathTraversal'] for events in
                        pd.read_csv(events_file_path, low_memory=False, chunksize=chunksize)])
    events['time'] = events['time'].astype('float')
    events = events.sort_values(by='time', ascending=True)

    hour2type2num_passenger = {}
    vehicle2passengers_and_type = {}
    last_hour = 0

    def update_last_hour_vehicles():
        cur_type2num_passenger = {}
        for _, (passengers, t) in vehicle2passengers_and_type.items():
            if t not in cur_type2num_passenger:
                cur_type2num_passenger[t] = 0
            cur_type2num_passenger[t] = cur_type2num_passenger[t] + passengers
        hour2type2num_passenger[last_hour] = cur_type2num_passenger

    for index, row in events.iterrows():
        hour = int(float(row['time']) / 3600)
        vehicle_type = row['vehicleType']
        v = row['vehicle']
        num_passengers = int(row['numPassengers'])
        if vehicle_type == 'BODY-TYPE-DEFAULT':
            continue
        if hour != last_hour:
            update_last_hour_vehicles()
            last_hour = hour
            vehicle2passengers_and_type = {}
        if (v not in vehicle2passengers_and_type) or (vehicle2passengers_and_type[v][0] < num_passengers):
            vehicle2passengers_and_type[v] = (num_passengers, vehicle_type)

    update_last_hour_vehicles()
    vehicles = set()
    for hour, data in hour2type2num_passenger.items():
        for v, _ in data.items():
            vehicles.add(v)

    hours = []
    res = {}
    for h, dataForHour in hour2type2num_passenger.items():
        hours.append(h)
        for v in vehicles:
            if v not in res:
                res[v] = []
            if v not in dataForHour:
                res[v].append(0)
            else:
                res[v].append(dataForHour[v])

    res['HOUR'] = hours
    rows = int(len(vehicles) / 2)

    fig1, axes = plt.subplots(rows, 2, figsize=(25, 7 * rows))
    fig1.tight_layout(pad=0.1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.1)
    res_df = pd.DataFrame(res)
    for i, v in enumerate(vehicles):
        if i < len(vehicles) - 1:
            res_df.plot(x='HOUR', y=v, ax=axes[int(i / 2)][i % 2])
        else:
            fig1, ax = plt.subplots(1, 1, figsize=(8, 7))
            fig1.tight_layout(pad=0.1)
            fig1.subplots_adjust(wspace=0.25, hspace=0.1)
            res_df.plot(x='HOUR', y=v, ax=ax)


def people_flow_in_cbd_s3(s3url, iteration):
    s3path = get_output_path_from_s3_url(s3url)
    events_file_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)
    return people_flow_in_cbd_file_path(events_file_path)


def people_flow_in_cbd_file_path(events_file_path, chunksize=100000):
    events = pd.concat([events[events['type'] == 'PathTraversal'] for events in
                        pd.read_csv(events_file_path, low_memory=False, chunksize=chunksize)])
    return people_flow_in_cdb(events)


def diff_people_flow_in_cbd_s3(s3url, iteration, s3url_base, iteration_base):
    s3path = get_output_path_from_s3_url(s3url)
    events_file_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)
    s3path_base = get_output_path_from_s3_url(s3url_base)
    events_file_path_base = s3path_base + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration_base)
    return diff_people_flow_in_cbd_file_path(events_file_path, events_file_path_base)


def diff_people_flow_in_cbd_file_path(events_file_path, events_file_path_base, chunksize=100000):
    events = pd.concat([events[events['type'] == 'PathTraversal'] for events in
                        pd.read_csv(events_file_path, low_memory=False, chunksize=chunksize)])
    events_base = pd.concat([events[events['type'] == 'PathTraversal'] for events in
                             pd.read_csv(events_file_path_base, low_memory=False, chunksize=chunksize)])
    return diff_people_in(events, events_base)


def people_flow_in_cdb(df):
    polygon = Polygon([
        (-74.005088, 40.779100),
        (-74.034957, 40.680314),
        (-73.968867, 40.717604),
        (-73.957924, 40.759091)
    ])

    def inside(x, y):
        point = Point(x, y)
        return polygon.contains(point)

    def num_people(row):
        mode = row['mode']
        if mode in ['walk', 'bike']:
            return 1
        elif mode == 'car':
            return 1 + row['numPassengers']
        else:
            return row['numPassengers']

    def benchmark():
        data = """mode,Entering,Leaving
subway,2241712,2241712
car,877978,877978
bus,279735,279735
rail,338449,338449
ferry,66932,66932
bike,33634,33634
tram,3528,3528
        """
        return pd.read_csv(StringIO(data)).set_index('mode')

    f = df[(df['type'] == 'PathTraversal')][['mode', 'numPassengers', 'startX', 'startY', 'endX', 'endY']].copy(
        deep=True)

    f['numPeople'] = f.apply(lambda row: num_people(row), axis=1)
    f = f[f['numPeople'] > 0]

    f['startIn'] = f.apply(lambda row: inside(row['startX'], row['startY']), axis=1)
    f['endIn'] = f.apply(lambda row: inside(row['endX'], row['endY']), axis=1)
    f['numIn'] = f.apply(lambda row: row['numPeople'] if not row['startIn'] and row['endIn'] else 0, axis=1)

    s = f.groupby('mode')[['numIn']].sum()
    b = benchmark()

    t = pd.concat([s, b], axis=1)
    t.fillna(0, inplace=True)

    t['percentIn'] = t['numIn'] * 100 / t['numIn'].sum()
    t['percent_ref'] = t['Entering'] * 100 / t['Entering'].sum()

    t = t[['numIn', 'Entering', 'percentIn', 'percent_ref']]

    t['diff'] = t['percentIn'] - t['percent_ref']
    t['diff'].plot(kind='bar', title="Diff: current - reference, %", figsize=(7, 5), legend=False, fontsize=12)

    t.loc["Total"] = t.sum()
    return t


def get_people_in(df):
    polygon = Polygon([
        (-74.005088, 40.779100),
        (-74.034957, 40.680314),
        (-73.968867, 40.717604),
        (-73.957924, 40.759091)
    ])

    def inside(x, y):
        point = Point(x, y)
        return polygon.contains(point)

    def num_people(row):
        mode = row['mode']
        if mode in ['walk', 'bike']:
            return 1
        elif mode == 'car':
            return 1 + row['numPassengers']
        else:
            return row['numPassengers']

    f = df[(df['type'] == 'PathTraversal') & (df['mode'].isin(['car', 'bus', 'subway']))][
        ['mode', 'numPassengers', 'startX', 'startY', 'endX', 'endY']].copy(deep=True)

    f['numPeople'] = f.apply(lambda row: num_people(row), axis=1)
    f = f[f['numPeople'] > 0]

    f['startIn'] = f.apply(lambda row: inside(row['startX'], row['startY']), axis=1)
    f['endIn'] = f.apply(lambda row: inside(row['endX'], row['endY']), axis=1)
    f['numIn'] = f.apply(lambda row: row['numPeople'] if not row['startIn'] and row['endIn'] else 0, axis=1)

    s = f.groupby('mode')[['numIn']].sum()

    s.fillna(0, inplace=True)

    s['percentIn'] = s['numIn'] * 100 / s['numIn'].sum()

    return s['percentIn']


def diff_people_in(current, base):
    def reference():
        data = """date,subway,bus,car
07/05/2020,-77.8,-35,-21.8
06/05/2020,-87.2,-64,-30.8
05/05/2020,-90.5,-73,-50.3
04/05/2020,-90.5,-71,-78.9
03/05/2020,0.0,4,-0.1
        """
        ref = pd.read_csv(StringIO(data), parse_dates=['date'])
        ref.sort_values('date', inplace=True)
        ref['month'] = ref['date'].dt.month_name()
        ref = ref.set_index('month').drop('date', 1)
        return ref

    b = get_people_in(base)
    c = get_people_in(current)
    b.name = 'base'
    c.name = 'current'

    t = pd.concat([b, c], axis=1)
    t['increase'] = t['current'] - t['base']

    pc = reference()

    run = t['increase'].to_frame().T
    run = run.reset_index().drop('index', 1)
    run['month'] = 'Run'
    run = run.set_index('month')
    result = pd.concat([run, pc], axis=0)

    result.plot(kind='bar', title="Diff current - reference, %", figsize=(10, 10), legend=True, fontsize=12)
    return result


def plot_hists(df, column_group_by, column_build_hist, ax, bins=100, alpha=0.2):
    for (i, d) in df.groupby(column_group_by):
        d[column_build_hist].hist(bins=bins, alpha=alpha, ax=ax, label=i)
    ax.legend()


def calc_number_of_rows_in_beamlog(s3url, keyword):
    s3path = get_output_path_from_s3_url(s3url)
    beamlog = urllib.request.urlopen(s3path + "/beamLog.out")
    count = 0
    for b_line in beamlog.readlines():
        line = b_line.decode("utf-8")
        if keyword in line:
            count = count + 1
    print("there are {} of '{}' in {}".format(count, keyword, s3path + '/beamLog.out'))


def grep_beamlog_for_errors_warnings(s3url):
    error_keywords = ["ERROR", "WARN"]
    error_patterns_for_count = [
        r".*StreetLayer - .* [0-9]*.*, skipping.*",
        r".*OsmToMATSim - Could not.*. Ignoring it.",
        r".*GeoUtilsImpl - .* Coordinate does not appear to be in WGS. No conversion will happen:.*",
        r".*InfluxDbSimulationMetricCollector - There are enabled metrics, but InfluxDB is unavailable at.*",
        r".*ClusterSystem-akka.*WARN.*PersonAgent.*didn't get nextActivity.*",
        r".*ClusterSystem-akka.*WARN.*Person Actor.*attempted to reserve ride with agent Actor.*"
        + "that was not found, message sent to dead letters.",
        r".*ClusterSystem-akka.*ERROR.*PersonAgent - State:FinishingModeChoice PersonAgent:[0-9]*[ ]*"
        + "Current tour vehicle is the same as the one being removed: [0-9]* - [0-9]*.*"
    ]

    error_count = {}
    for error in error_patterns_for_count:
        error_count[error] = 0

    print("")
    print("UNEXPECTED errors | warnings:")
    print("")

    s3path = get_output_path_from_s3_url(s3url)
    file = urllib.request.urlopen(s3path + "/beamLog.out")
    for b_line in file.readlines():
        line = b_line.decode("utf-8")

        found = False
        for error_pattern in error_patterns_for_count:
            matched = re.match(error_pattern, line)
            if bool(matched):
                found = True
                error_count[error_pattern] = error_count[error_pattern] + 1

        if found:
            continue

        for error in error_keywords:
            if error in line:
                print(line)

    print("")
    print("expected errors | warnings:")
    print("")
    for error, count in error_count.items():
        print(count, "of", error)


def get_default_and_emergency_parkings(s3url, iteration):
    s3path = get_output_path_from_s3_url(s3url)
    parking_file_path = s3path + "/ITERS/it.{0}/{0}.parkingStats.csv".format(iteration)
    parking_df = pd.read_csv(parking_file_path)
    parking_df['TAZ'] = parking_df['TAZ'].astype(str)
    filtered_df = parking_df[
        (parking_df['TAZ'].str.contains('default')) | (parking_df['TAZ'].str.contains('emergency'))]
    res_df = filtered_df.groupby(['TAZ']).count().reset_index()[['TAZ', 'timeBin']] \
        .rename(columns={'timeBin': 'count'})
    return res_df


def load_modechoices(events_file_path, chunksize=100000):
    start_time = time.time()
    df = pd.concat(
        [df[df['type'] == 'ModeChoice'] for df in pd.read_csv(events_file_path, low_memory=False, chunksize=chunksize)])
    print("events file url:", events_file_path)
    print("modechoice loading took %s seconds" % (time.time() - start_time))
    return df


def analyze_mode_choice_changes(title_to_s3url, benchmark_url):
    # def get_realized_modes(s3url, data_file_name='referenceRealizedModeChoice.csv'):
    def get_realized_modes(s3url, data_file_name='realizedModeChoice.csv'):
        modes = ['bike', 'car', 'cav', 'drive_transit', 'ride_hail',
                 'ride_hail_pooled', 'ride_hail_transit', 'walk', 'walk_transit']

        path = get_output_path_from_s3_url(s3url) + "/" + data_file_name
        df = pd.read_csv(path, names=modes)
        tail = df.tail(1).copy()

        for mode in modes:
            tail[mode] = tail[mode].astype(float)

        return tail

    benchmark = get_realized_modes(benchmark_url).reset_index(drop=True)

    modechoices_difference = []
    modechoices_diff_in_percentage = []

    for (name, url) in title_to_s3url:
        modechoice = get_realized_modes(url).reset_index(drop=True)
        modechoice = modechoice.sub(benchmark, fill_value=0)
        modechoice_perc = modechoice / benchmark * 100

        modechoice['name'] = name
        modechoice['sim_url'] = url
        modechoices_difference.append(modechoice)

        modechoice_perc['name'] = name
        modechoice_perc['sim_url'] = url
        modechoices_diff_in_percentage.append(modechoice_perc)

    df_diff = pd.concat(modechoices_difference)
    df_diff_perc = pd.concat(modechoices_diff_in_percentage)

    _, (ax1, ax2) = plt.subplots(2, 1, sharex='all', figsize=(20, 8))

    df_diff.set_index('name').plot(kind='bar', ax=ax1, rot=65)
    df_diff_perc.set_index('name').plot(kind='bar', ax=ax2, rot=65)

    ax1.axhline(0, color='black', linewidth=0.4)
    ax1.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax1.set_title('difference between run and benchmark in absolute numbers')
    ax1.grid('on', which='major', axis='y')

    ax2.axhline(0, color='black', linewidth=0.4)
    ax2.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax2.set_title('difference between run and benchmark in percentage')
    ax2.grid('on', which='major', axis='y')

    plt.suptitle("BEAM run minus benchmark run. realizedModeChoice.csv")
    return benchmark


def load_activities(events_file_path, chunksize=100000):
    start_time = time.time()
    df = pd.concat(
        [df[(df['type'] == 'actstart') | (df['type'] == 'actend')] for df in
         pd.read_csv(events_file_path, low_memory=False, chunksize=chunksize)])
    df['hour'] = (df['time'] / 3600).astype(int)
    print("events file url:", events_file_path)
    print("actstart and actend events loading took %s seconds" % (time.time() - start_time))
    return df


def plot_modechoice_distance_distribution(s3url, iteration):
    s3path = get_output_path_from_s3_url(s3url)
    events_file_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)

    start_time = time.time()
    events_file = pd.concat([df[df['type'] == 'ModeChoice']
                             for df in pd.read_csv(events_file_path, low_memory=False, chunksize=100000)])
    print("modechoice loading took %s seconds" % (time.time() - start_time))

    events_file['length'].hist(bins=100, by=events_file['mode'], figsize=(20, 12), rot=10, sharex=True)


def get_average_car_speed(s3url, iteration):
    s3path = get_output_path_from_s3_url(s3url)
    average_speed = pd.read_csv(s3path + "/AverageCarSpeed.csv")
    return average_speed[average_speed['iteration'] == iteration]['speed'].median()


def plot_simulation_volumes_vs_bench(s3url, iteration, ax, title="Volume SUM comparison with benchmark.",
                                     simulation_volumes=None, s3path=None):
    if s3path is None:
        s3path = get_output_path_from_s3_url(s3url)

    def calc_sum_of_link_stats(link_stats_file_path, chunksize=100000):
        start_time = time.time()
        df = pd.concat([df.groupby('hour')['volume'].sum() for df in
                        pd.read_csv(link_stats_file_path, low_memory=False, chunksize=chunksize)])
        df = df.groupby('hour').sum().to_frame(name='sum')
        # print("link stats url:", link_stats_file_path)
        print("link stats downloading and calculation took %s seconds" % (time.time() - start_time))
        return df

    if simulation_volumes is None:
        linkstats_path = s3path + "/ITERS/it.{0}/{0}.linkstats.csv.gz".format(iteration)
        simulation_volumes = calc_sum_of_link_stats(linkstats_path)

    color_benchmark = 'tab:red'
    color_volume = 'tab:green'

    ax1 = ax

    ax1.set_title('{} iter {}'.format(title, iteration))
    ax1.set_xlabel('hour of day')

    ax1.plot(range(0, 24), nyc_volumes_benchmark['count'], color=color_benchmark, label="benchmark")
    ax1.plot(np.nan, color=color_volume, label="simulation volume")  # to have both legends on same axis
    ax1.legend(loc="upper right")
    ax1.xaxis.set_ticks(np.arange(0, 24, 1))

    ax1.tick_params(axis='y', labelcolor=color_benchmark)

    volume_per_hour = simulation_volumes[0:23]['sum']
    volume_hours = list(volume_per_hour.index)

    shifted_hours = list(map(lambda x: x + 1, volume_hours))

    ax12 = ax1.twinx()  # to plot things on the same graph but with different Y axis
    ax12.plot(shifted_hours, volume_per_hour, color=color_volume)
    ax12.tick_params(axis='y', labelcolor=color_volume)

    return simulation_volumes


def plot_activities_ends_vs_bench(s3url, iteration, ax, ax2=None, title="Activity ends comparison.", population_size=1,
                                  activity_ends=None, s3path=None):
    if s3path is None:
        s3path = get_output_path_from_s3_url(s3url)

    def load_activity_ends(events_file_path, chunksize=100000):
        start_time = time.time()
        try:
            df = pd.concat([df[df['type'] == 'actend']
                            for df in pd.read_csv(events_file_path, low_memory=False, chunksize=chunksize)])
        except HTTPError:
            raise NameError('can not download file by url:', events_file_path)
        df['hour'] = (df['time'] / 3600).astype(int)
        # print("events file url:", events_file_path)
        print("activity ends loading took %s seconds" % (time.time() - start_time))
        return df

    if activity_ends is None:
        events_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)
        activity_ends = load_activity_ends(events_path)

    color_act_ends = 'tab:blue'

    ax.set_title('{} iter {} [{} total act ends]'.format(title, iteration, activity_ends.shape[0]))
    ax.set_xlabel('hour of day')
    ax.xaxis.set_ticks(np.arange(0, 24, 1))

    act_ends_24 = activity_ends[activity_ends['hour'] <= 24].copy()

    act_ends_total = act_ends_24.groupby('hour')['hour'].count() / population_size
    act_ends_hours = list(act_ends_total.index)

    def plot_act_ends(ax_to_plot, act_type):
        df = act_ends_24[act_ends_24['actType'] == act_type].groupby('hour')['hour'].count() / population_size
        ax_to_plot.plot(df.index, df, label='# of {} ends'.format(act_type))

    def plot_benchmark_and_legend(ax_to_plot):
        color_benchmark = 'black'
        ax_to_plot.plot(np.nan, color=color_benchmark,
                        label='benchmark (right scale)')  # to have both legends on same axis

        ax_to_plot.legend(loc="upper right")
        ax_to_plot.tick_params(axis='y', labelcolor=color_act_ends)

        ax_twinx = ax_to_plot.twinx()  # to plot things on the same graph but with different Y axis
        ax_twinx.plot(range(0, 24), nyc_activity_ends_benchmark, color=color_benchmark)
        ax_twinx.tick_params(axis='y', labelcolor=color_benchmark)

    ax.plot(act_ends_hours, act_ends_total, color=color_act_ends, label='# of activity ends', linewidth=3)
    plot_act_ends(ax, 'Work')
    plot_act_ends(ax, 'Home')

    plot_benchmark_and_legend(ax)

    if ax2 is not None:
        ax2.set_title('other activities')
        ax2.set_xlabel('hour of day')
        ax2.xaxis.set_ticks(np.arange(0, 24, 1))

        plot_act_ends(ax2, 'Meal')
        plot_act_ends(ax2, 'SocRec')
        plot_act_ends(ax2, 'Shopping')
        plot_act_ends(ax2, 'Other')

        plot_benchmark_and_legend(ax2)

    return activity_ends


def plot_volumes_comparison_on_axs(s3url, iteration, suptitle="", population_size=1,
                                   simulation_volumes=None, activity_ends=None,
                                   plot_simulation_volumes=True, plot_activities_ends=True):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 7))
    fig1.tight_layout(pad=0.1)
    fig1.subplots_adjust(wspace=0.25, hspace=0.1)
    plt.xticks(np.arange(0, 24, 2))
    plt.suptitle(suptitle, y=1.05, fontsize=17)

    if plot_simulation_volumes:
        plot_simulation_volumes_vs_bench(s3url, iteration=iteration, ax=ax1,
                                         title="Volume SUM comparison with benchmark.",
                                         simulation_volumes=simulation_volumes)

    if plot_activities_ends:
        plot_activities_ends_vs_bench(s3url, iteration=iteration, ax=ax2, title="Activity ends comparison.",
                                      population_size=population_size, activity_ends=activity_ends)


def analyze_fake_walkers(s3url, iteration, threshold=2000, title="", modechoice=None):
    s3path = get_output_path_from_s3_url(s3url)
    events_file_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)

    if modechoice is None:
        modechoice = load_modechoices(events_file_path)

    is_fake = (modechoice['length'] >= threshold) & (
            (modechoice['availableAlternatives'] == 'WALK') | (modechoice['availableAlternatives'].isnull()))

    fake_walkers = modechoice[(modechoice['mode'] == 'walk') & is_fake]
    real_walkers = modechoice[(modechoice['mode'] == 'walk') & (~is_fake)]

    fig, axs = plt.subplots(2, 2, figsize=(24, 4 * 2))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.suptitle(title, y=1.11)

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]

    fake_walkers['length'].hist(bins=200, ax=ax1, alpha=0.3, label='fake walkers')
    real_walkers['length'].hist(bins=200, ax=ax1, alpha=0.3, label='real walkers')
    ax1.legend(loc='upper right', prop={'size': 10})
    ax1.set_title("Trip length histogram. Fake vs Real walkers. Min length of trip is {0}".format(threshold))
    ax1.axvline(5000, color="black", linestyle="--")

    fake_walkers['length'].hist(bins=200, ax=ax2, log=True, alpha=0.3, label='fake walkers')
    real_walkers['length'].hist(bins=200, ax=ax2, log=True, alpha=0.3, label='real walkers')
    ax2.legend(loc='upper right', prop={'size': 10})
    ax2.set_title(
        "Trip length histogram. Fake vs Real walkers. Logarithmic scale. Min length of trip is {0}".format(threshold))
    ax2.axvline(5000, color="black", linestyle="--")

    ax1 = axs[1, 0]
    ax2 = axs[1, 1]

    long_real_walkers = real_walkers[real_walkers['length'] >= threshold]
    number_of_top_alternatives = 5
    walkers_by_alternative = long_real_walkers.groupby('availableAlternatives')['length'].count().sort_values(
        ascending=False)
    top_alternatives = set(
        walkers_by_alternative.reset_index()['availableAlternatives'].head(number_of_top_alternatives))

    for alternative in top_alternatives:
        selected = long_real_walkers[long_real_walkers['availableAlternatives'] == alternative]['length']
        selected.hist(bins=200, ax=ax1, alpha=0.4, linewidth=4, label=alternative)
        selected.hist(bins=20, ax=ax2, log=True, histtype='step', linewidth=4, label=alternative)

    ax1.set_title("Length histogram of top {} alternatives of real walkers".format(number_of_top_alternatives))
    ax1.legend(loc='upper right', prop={'size': 10})
    ax2.set_title(
        "Length histogram of top {} alternatives of real walkers. Logarithmic scale".format(number_of_top_alternatives))
    ax2.legend(loc='upper right', prop={'size': 10})

    number_of_fake_walkers = fake_walkers.shape[0]
    number_of_real_walkers = real_walkers.shape[0]
    number_of_all_modechoice = modechoice.shape[0]

    print('number of all modechoice events', number_of_all_modechoice)
    print('number of real walkers, real walkers of all modechoice events :')
    print(number_of_real_walkers, number_of_real_walkers / number_of_all_modechoice)
    print('number of FAKE walkers, FAKE walkers of all modechoice events :')
    print(number_of_fake_walkers, number_of_fake_walkers / number_of_all_modechoice)

    return "{},{},{},{},{}".format(number_of_real_walkers, number_of_real_walkers / number_of_all_modechoice,
                                   number_of_fake_walkers, number_of_fake_walkers / number_of_all_modechoice,
                                   number_of_all_modechoice)


def parse_config(config_url, complain=True):
    config = urllib.request.urlopen(config_url)

    config_keys = ["flowCapacityFactor", "speedScalingFactor", "quick_fix_minCarSpeedInMetersPerSecond",
                   "activitySimEnabled", "transitCapacity",
                   "minimumRoadSpeedInMetersPerSecond", "fractionOfInitialVehicleFleet",
                   "agentSampleSizeAsFractionOfPopulation",
                   "simulationName", "directory", "generate_secondary_activities", "lastIteration",
                   "fractionOfPeopleWithBicycle",
                   "parkingStallCountScalingFactor", "parkingPriceMultiplier", "parkingCostScalingFactor", "queryDate",
                   "transitPrice", "transit_crowding", "transit_crowding_percentile",
                   "maxLinkLengthToApplySpeedScalingFactor", "max_destination_distance_meters",
                   "max_destination_choice_set_size",
                   "transit_crowding_VOT_multiplier", "transit_crowding_VOT_threshold",
                   "activity_file_path", "intercept_file_path", "additional_trip_utility",
                   "ModuleProbability_1", "ModuleProbability_2", "ModuleProbability_3", "ModuleProbability_4"]
    intercept_keys = ["bike_intercept", "car_intercept", "drive_transit_intercept", "ride_hail_intercept",
                      "ride_hail_pooled_intercept", "ride_hail_transit_intercept", "walk_intercept",
                      "walk_transit_intercept", "transfer"]

    config_map = {}
    default_value = ""

    for conf_key in config_keys:
        config_map[conf_key] = default_value

    def set_value(key, line_value):
        value = line_value.strip().replace("\"", "")

        if key not in config_map:
            config_map[key] = value
        else:
            old_val = config_map[key]
            if old_val == default_value or old_val.strip() == value.strip():
                config_map[key] = value
            else:
                if complain:
                    print("an attempt to rewrite config value with key:", key)
                    print("   value in the map  \t", old_val)
                    print("   new rejected value\t", value)

    physsim_names = ['JDEQSim', 'BPRSim', 'PARBPRSim', 'CCHRoutingAssignment']

    def look_for_physsim_type(config_line):
        for physsim_name in physsim_names:
            if 'name={}'.format(physsim_name) in config_line:
                set_value("physsim_type", "physsim_type = {}".format(physsim_name))

    for b_line in config.readlines():
        line = b_line.decode("utf-8").strip()

        look_for_physsim_type(line)

        for ckey in config_keys:
            if ckey + "=" in line or ckey + "\"=" in line:
                set_value(ckey, line)

        for ikey in intercept_keys:
            if ikey in line:
                set_value(ikey, line)

    return config_map


def get_calibration_text_data(s3url, commit=""):
    print("order: car | walk | bike | ride_hail | ride_hail_transit | walk_transit | drive_transit | ride_hail_pooled")
    print("")

    print('ordered realized mode choice:')
    print('ordered commute realized mode choice:')
    modes_section = get_realized_modes_as_str(s3url)
    print(modes_section)
    print(get_realized_modes_as_str(s3url, 'referenceRealizedModeChoice_commute.csv'))
    print("")

    s3path = get_output_path_from_s3_url(s3url)
    config = parse_config(s3path + "/fullBeamConfig.conf")

    def get_config_value(conf_value_name):
        return config.get(conf_value_name, '=default').split('=')[-1]

    intercepts = ["car_intercept", "walk_intercept", "bike_intercept", "ride_hail_intercept",
                  "ride_hail_transit_intercept",
                  "walk_transit_intercept", "drive_transit_intercept", "ride_hail_pooled_intercept", "transfer"]
    print('order of intercepts:', "\n\t\t ".join(intercepts))
    intercepts_sections = ', '.join(get_config_value(x) for x in intercepts)
    print(intercepts_sections)
    print("")

    config_ordered = ["lastIteration", "agentSampleSizeAsFractionOfPopulation", "flowCapacityFactor",
                      "speedScalingFactor",
                      "quick_fix_minCarSpeedInMetersPerSecond", "minimumRoadSpeedInMetersPerSecond",
                      "fractionOfInitialVehicleFleet", "transitCapacity", "fractionOfPeopleWithBicycle",
                      "parkingStallCountScalingFactor", "transitPrice", "transit_crowding_VOT_multiplier",
                      "transit_crowding_VOT_threshold", "additional_trip_utility"]
    print('order of config values:', "\n\t\t ".join(config_ordered))
    config_section = ','.join(get_config_value(x) for x in config_ordered)
    print(config_section)
    print("")

    print('the rest of configuration:')
    for key, value in config.items():
        if 'intercept' not in key and key not in config_ordered:
            print(value)

    print("")
    grep_beamlog(s3path + "/beamLog.out", ["Total number of links", "Number of persons:"])

    return "{}, ,{},{}, , ,{}, ,{}".format(config_section, commit, s3url, modes_section, intercepts_sections)


def calculate_median_time_at_home(s3url, iteration, total_persons, debug_print=False):
    s3path = get_output_path_from_s3_url(s3url)
    events_file_path = s3path + "/ITERS/it.{0}/{0}.events.csv.gz".format(iteration)

    home_acts = pd.concat([events[events['actType'] == 'Home']
                           for events in pd.read_csv(events_file_path, low_memory=False, chunksize=10000)])

    def get_home_activity_time(row):
        if row['type'] == 'actend':
            return min(row['time'] / 3600, 24.0)
        if row['type'] == 'actstart':
            return max(row['time'] / -3600, -23.9)
        return 0

    home_acts['homeActTime'] = home_acts.apply(get_home_activity_time, axis=1)
    home_activities = ((home_acts.groupby('person')['homeActTime']).sum() + 24).reset_index()

    affected_persons = len(home_acts['person'].unique())

    all_people_home_time = list(home_activities['homeActTime']) + [24] * (total_persons - affected_persons)
    median_time_at_home = statistics.median(all_people_home_time)
    if debug_print:
        print('all people home time. len:{} sum:{} mean:{} median:{}'.format(len(all_people_home_time),
                                                                             sum(all_people_home_time),
                                                                             sum(all_people_home_time) / len(
                                                                                 all_people_home_time),
                                                                             median_time_at_home))

    return median_time_at_home


def plot_median_time_at_home(title_to_s3url, total_persons, iteration, figsize=(30, 5)):
    mean_time = []

    for ((title, s3url), ax_idx) in zip(title_to_s3url, range(len(title_to_s3url))):
        median_time = calculate_median_time_at_home(s3url, total_persons=total_persons, iteration=iteration)
        mean_time.append((title, median_time))

    baseline = mean_time[0][1]

    time_at_home_vs_baseline = ([], [])

    for (title, avg_time_at_home) in mean_time:
        ratio = avg_time_at_home / baseline
        time_at_home_vs_baseline[0].append(title)
        time_at_home_vs_baseline[1].append(ratio)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    x = range(len(mean_time))
    y = time_at_home_vs_baseline[1]
    plt.xticks(x, time_at_home_vs_baseline[0])
    ax.plot(x, y)
    ax.set_title("Median time at home months vs baseline")


def compare_riderships_vs_baserun_and_benchmark(title_to_s3url, iteration, s3url_base_run, date_to_calc_diff=None,
                                                figsize=(20, 5), rot=15, suptitle="",
                                                plot_columns=None, plot_reference=True):
    columns = ['date', 'subway', 'bus', 'rail', 'car', 'transit']

    benchmark_mta_info = [['09 2020\n  mta.info', -72.90, -54.00, -78.86, -12.90, -68.42],
                          ['08 2020\n  mta.info', -75.50, -40.00, -83.32, -08.90, -66.68],
                          ['07 2020\n  mta.info', -79.60, -49.00, -83.91, -16.20, -71.90],
                          ['06 2020\n  mta.info', -87.60, -66.00, -90.95, -37.40, -82.17],
                          ['05 2020\n  mta.info', -90.70, -75.00, -95.00, -52.30, -86.89],
                          ['04 2020\n  mta.info', -90.60, -77.00, -96.13, -63.20, -87.47]]

    if not plot_columns:
        plot_columns = columns[1:]

    date_to_benchmark = {}
    for row in benchmark_mta_info:
        date_to_benchmark[row[0]] = row[1:]

    print('reference dates:', date_to_benchmark.keys())

    def column_name_to_passenger_multiplier(column_name):
        if column_name == '0':
            return 1

        delimeter = '-'
        if delimeter in column_name:
            nums = column_name.split(delimeter)
            return (int(nums[0]) + int(nums[1])) // 2
        else:
            return int(column_name)

    def get_sum_of_passenger_per_trip(df, ignore_hour_0=True):
        sum_df = df.sum()
        total_sum = 0

        for column in df.columns:
            if column == 'hours':
                continue
            if ignore_hour_0 and column == '0':
                continue
            multiplier = column_name_to_passenger_multiplier(column)
            total_sum = total_sum + sum_df[column] * multiplier

        return total_sum

    def get_car_bus_subway_trips(beam_s3url):
        s3path = get_output_path_from_s3_url(beam_s3url)

        def read_csv(filename):
            file_url = s3path + "/ITERS/it.{0}/{0}.{1}.csv".format(iteration, filename)
            try:
                return pd.read_csv(file_url)
            except HTTPError:
                print('was not able to download', file_url)

        sub_trips = read_csv('passengerPerTripSubway')
        bus_trips = read_csv('passengerPerTripBus')
        car_trips = read_csv('passengerPerTripCar')
        rail_trips = read_csv('passengerPerTripRail')

        sub_trips_sum = get_sum_of_passenger_per_trip(sub_trips, ignore_hour_0=True)
        bus_trips_sum = get_sum_of_passenger_per_trip(bus_trips, ignore_hour_0=True)
        car_trips_sum = get_sum_of_passenger_per_trip(car_trips, ignore_hour_0=False)
        rail_trips_sum = get_sum_of_passenger_per_trip(rail_trips, ignore_hour_0=True)

        return car_trips_sum, bus_trips_sum, sub_trips_sum, rail_trips_sum

    (base_car, base_bus, base_sub, base_rail) = get_car_bus_subway_trips(s3url_base_run)

    graph_data = []

    for (run_title, s3url_run) in title_to_s3url:
        (minus_car, minus_bus, minus_sub, minus_rail) = get_car_bus_subway_trips(s3url_run)

        def calc_diff(base_run_val, minus_run_val):
            return (minus_run_val - base_run_val) / base_run_val * 100

        diff_transit = calc_diff(base_sub + base_bus + base_rail, minus_sub + minus_bus + minus_rail)
        diff_sub = calc_diff(base_sub, minus_sub)
        diff_bus = calc_diff(base_bus, minus_bus)
        diff_car = calc_diff(base_car, minus_car)
        diff_rail = calc_diff(base_rail, minus_rail)

        graph_data.append(['{0}'.format(run_title), diff_sub, diff_bus, diff_rail, diff_car, diff_transit])

    def plot_bars(df, ax, title, columns_to_plot):
        df.groupby('date').sum()[columns_to_plot].plot(kind='bar', ax=ax, rot=rot)
        ax.grid('on', which='major', axis='y')
        ax.set_title(title)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))

    if date_to_calc_diff:
        fig, axs = plt.subplots(1, 2, sharey='all', figsize=figsize)
        ax_main = axs[0]
    else:
        fig, axs = plt.subplots(1, 1, sharey='all', figsize=figsize)
        ax_main = axs

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(wspace=0.25, hspace=0.1)

    plt.suptitle('Comparison of difference vs baseline and vs real data from MTI.info\n{}'.format(suptitle), y=1.2,
                 fontsize=17)

    result = pd.DataFrame(graph_data, columns=columns)
    if plot_reference:
        reference_df = pd.DataFrame(benchmark_mta_info, columns=columns)
        result = result.append(reference_df)

    plot_bars(result, ax_main, 'reference from mta.info vs BEAM simulation', plot_columns)

    if date_to_calc_diff:
        diff = result[columns[1:]].sub(date_to_benchmark[date_to_calc_diff], axis=1)
        diff[columns[0]] = result[columns[0]]
        plot_bars(diff, axs[1], 'result minus reference', plot_columns)


def plot_modechoice_comparison(title_to_s3url, benchmark_url, benchmark_name="benchmark"):
    modes = ['bike', 'car', 'drive_transit', 'ride_hail',
             'ride_hail_pooled', 'ride_hail_transit', 'walk', 'walk_transit']

    # def get_realized_modes(s3url, data_file_name='referenceRealizedModeChoice.csv'):
    def get_realized_modes(s3url, data_file_name='realizedModeChoice.csv'):
        path = get_output_path_from_s3_url(s3url) + "/" + data_file_name
        df = pd.read_csv(path)
        tail = df.tail(1).copy()

        exist_columns = set(tail.columns)
        for m in modes:
            if m not in exist_columns:
                tail[m] = 0.0
            else:
                tail[m] = tail[m].astype(float)

        return tail[modes]

    benchmark = get_realized_modes(benchmark_url).reset_index(drop=True)

    benchmark_absolute = benchmark.copy()
    benchmark_absolute['name'] = benchmark_name

    zeros = benchmark_absolute.copy()
    for mode in modes:
        zeros[mode] = 0.0

    modechoices_absolute = [benchmark_absolute]
    modechoices_difference = [zeros]
    modechoices_diff_in_percentage = [zeros]

    for (name, url) in title_to_s3url:
        modechoice = get_realized_modes(url).reset_index(drop=True)

        modechoice_absolute = modechoice.copy()
        modechoice_absolute['name'] = name
        modechoices_absolute.append(modechoice_absolute)

        modechoice = modechoice.sub(benchmark, fill_value=0)
        modechoice_perc = modechoice / benchmark * 100

        modechoice['name'] = name
        modechoices_difference.append(modechoice)

        modechoice_perc['name'] = name
        modechoices_diff_in_percentage.append(modechoice_perc)

    df_absolute = pd.concat(modechoices_absolute)
    df_diff = pd.concat(modechoices_difference)
    df_diff_perc = pd.concat(modechoices_diff_in_percentage)

    _, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex='all', figsize=(20, 5 * 3))

    def plot(df, ax, title):
        df.set_index('name').plot(kind='bar', ax=ax, rot=65)
        ax.axhline(0, color='black', linewidth=0.4)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_title(title)

    plot(df_absolute, ax0, "absolute values of modechoice")
    plot(df_diff, ax1, "difference between run and benchmark in absolute numbers")
    plot(df_diff_perc, ax2, "difference between run and benchmark in percentage")

    plt.suptitle("BEAM run vs benchmark. realizedModeChoice.csv")


def load_mapping():
    return pd.read_csv("https://github.com/LBNL-UCB-STI/beam/files/5146939/beam_transcom_mapping.csv.gz")


def load_tmc_dictionary():
    mapping = load_mapping()

    tmc_path = "https://beam-outputs.s3.amazonaws.com/new_city/newyork/DOT_Traffic_Speeds_20200301.csv.gz"
    tmc_df = pd.concat([df[df['LINK_ID'].isin(mapping['trafLink'])]
                        for df in
                        pd.read_csv(tmc_path, low_memory=False, chunksize=100000, parse_dates=['DATA_AS_OF'])])
    wed = tmc_df[(tmc_df['DATA_AS_OF'].dt.dayofweek == 2)].copy()

    def group_speed_by_hour(tmc_original):
        tmc = tmc_original[(tmc_original['SPEED'] > 0) & (tmc_original['SPEED'] < 100)].copy()
        tmc['hour'] = tmc['DATA_AS_OF'].dt.hour
        to_plot = tmc[['hour', 'SPEED']].groupby(['hour']).mean()
        return to_plot

    months = wed['DATA_AS_OF'].dt.month.unique()
    return {month: group_speed_by_hour(wed[wed['DATA_AS_OF'].dt.month == month]) for month in months}


def plot_link_graphs(tmc_data, s3url, iteration, ax=None):
    mapping = load_mapping()

    s3path = get_output_path_from_s3_url(s3url)
    linkstats_path = f"{s3path}/ITERS/it.{iteration}/{iteration}.linkstats.csv.gz"
    ls = pd.concat([df[df['link'].isin(mapping['beamLink'])] for df in pd.read_csv(linkstats_path, chunksize=100000)])

    ms_to_mph = 2.23694
    ls['speed'] = ms_to_mph * ls['length'] / ls['traveltime']

    ax = tmc_data.plot(y='SPEED', label='transcom', ax=ax)

    beam_plot = ls[['hour', 'speed', 'volume']][ls['volume'] > 0].groupby('hour').apply(
        lambda x: np.average(x['speed'], weights=x['volume']))
    beam_plot = beam_plot[beam_plot.index < 24]
    beam_plot.plot(y='speed', label='beam', ax=ax)

    ax.set_ylabel("speed MPH")

    return ax


def plot_calibration_parameters(title_to_s3url,
                                suptitle="", figsize=(23, 6), rot=70,
                                calibration_parameters=None,
                                removal_probabilities=None):
    if calibration_parameters is None:
        calibration_parameters = ['additional_trip_utility', 'walk_transit_intercept']

    calibration_values = []

    for (title, s3url) in title_to_s3url:
        s3path = get_output_path_from_s3_url(s3url)
        config = parse_config(s3path + "/fullBeamConfig.conf", complain=False)

        def get_config_value(conf_value_name):
            return config.get(conf_value_name, '=default').split('=')[-1]

        param_values = [title]
        for param in calibration_parameters:
            param_value = get_config_value(param)
            param_values.append(float(param_value))

        calibration_values.append(param_values)

    calibration_parameters.insert(0, 'name')
    result = pd.DataFrame(calibration_values, columns=calibration_parameters)

    linewidth = 4
    removal_probabilities_color = 'green'

    ax = result.plot(x='name', figsize=figsize, rot=rot, linewidth=linewidth)

    # for (idx, params) in zip(range(len(calibration_values)), calibration_values):
    #     for param in params[1:]:
    #         plt.annotate(param, (idx, param))  # , textcoords="offset points", xytext=(0,10), ha='center')

    if removal_probabilities:
        ax.plot(np.NaN, label='removal probabilities (right scale)',
                color=removal_probabilities_color, linewidth=linewidth)

    ax.set_title('calibration parameters {}'.format(suptitle))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.grid('on', which='major', axis='y')

    if removal_probabilities:
        ax2 = ax.twinx()
        ax2.plot(range(len(removal_probabilities)), removal_probabilities,
                 color=removal_probabilities_color, alpha=0.5, linewidth=linewidth)


nyc_volumes_benchmark_date = '2018-04-11'
nyc_volumes_benchmark_raw = read_traffic_counts(
    pd.read_csv('https://data.cityofnewyork.us/api/views/ertz-hr4r/rows.csv?accessType=DOWNLOAD'))
nyc_volumes_benchmark = aggregate_per_hour(nyc_volumes_benchmark_raw, nyc_volumes_benchmark_date)

# from Zach
# index is hour
nyc_activity_ends_benchmark = [0.010526809, 0.007105842, 0.003006647, 0.000310397, 0.011508960, 0.039378258,
                               0.116178879, 0.300608907, 0.301269741, 0.214196234, 0.220456846, 0.237608230,
                               0.258382041, 0.277933413, 0.281891163, 0.308248524, 0.289517677, 0.333402259,
                               0.221353890, 0.140322664, 0.110115403, 0.068543370, 0.057286657, 0.011845660]

print("initialized")
