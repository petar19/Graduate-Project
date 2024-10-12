
import math
def calculate_distance_haversine(lat1,lon1,lat2,lon2):
    # approximate radius of earth in km
    R = 6373000

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c

    return distance

# time is "hh:mm:ss"
def time_to_s(time):
    split = time.split(":")
    return 3600 * int(split[0]) + 60 * int(split[1]) + int(split[2])

def calculate_time_diff(time1, time2):
    time1_s = time_to_s(time1)
    time2_s = time_to_s(time2)

    diff = time2_s - time1_s

    if diff < 0: # if next day
        diff += 86400

    return diff

import pandas as pd
import csv
import numpy as np

def separate_routes():
    routes = pd.read_csv("original_zet/routes.txt", quoting=csv.QUOTE_NONNUMERIC)
    routes.drop(["route_short_name","agency_id","route_desc","route_type","route_url","route_color","route_text_color"], axis=1, inplace=True)
    routes["route_id"] = routes["route_id"].apply(np.ushort)
    tram_routes = routes[routes["route_id"] < 100]
    bus_routes = routes[routes["route_id"] >= 100]
    tram_routes.to_csv("zet_processed/tram_routes.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    bus_routes.to_csv("zet_processed/bus_routes.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

def separate_trips():
    trips = pd.read_csv("original_zet/trips.txt", quoting=csv.QUOTE_NONNUMERIC)
    trips.drop(["trip_short_name","shape_id"], axis=1, inplace=True)
    trips = trips.sort_values(by=["route_id", "block_id"])
    trips[["route_id","direction_id","block_id"]] = trips[["route_id","direction_id","block_id"]].applymap(np.ushort)
    tram_trips = trips[trips["route_id"] < 100]
    bus_trips = trips[trips["route_id"] >= 100]
    tram_trips.to_csv("zet_processed/tram_trips.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    bus_trips.to_csv("zet_processed/bus_trips.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

def separate_stop_times():
    stop_times = pd.read_csv("original_zet/stop_times.txt", quoting=csv.QUOTE_NONNUMERIC)
    stop_times.drop(["arrival_time","pickup_type","drop_off_type","shape_dist_traveled"], axis=1, inplace=True)
    stop_times = stop_times.rename(columns = {"departure_time":"time"})
    stop_times[["stop_id","stop_sequence"]] = stop_times[["stop_id","stop_sequence"]].applymap(np.ushort)
    stop_times["service_type"] = stop_times.apply(lambda row : int(row["trip_id"].split("_")[1]),axis=1)
    stop_times["vehicle_id"] = stop_times.apply(lambda row : int(row["trip_id"].split("_")[3]),axis=1)
    tram_stop_times = stop_times[stop_times["vehicle_id"] < 100]
    bus_stop_times = stop_times[stop_times["vehicle_id"] >= 100]
    tram_stop_times.to_csv("zet_processed/tram_stop_times.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    bus_stop_times.to_csv("zet_processed/bus_stop_times.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    return tram_stop_times

def separate_stops(tram_stops_list):
    stops = pd.read_csv("original_zet/stops.txt", quoting=csv.QUOTE_NONNUMERIC)
    stops.drop(["stop_code","stop_desc","zone_id","stop_url","location_type","parent_station"], axis=1, inplace=True)
    stops["stop_id"] = stops["stop_id"].apply(np.ushort)
    tram_stops = stops[stops["stop_id"].isin(tram_stops_list)]
    bus_stops = stops[~stops["stop_id"].isin(tram_stops_list)]
    tram_stops.to_csv("zet_processed/tram_stops.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    bus_stops.to_csv("zet_processed/bus_stops.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

'''
4 part method that separates bus and tram data:
'''
def separate_bus_tram():
    separate_routes()
    separate_trips()
    tram_stop_times = separate_stop_times()
    separate_stops(list(tram_stop_times["stop_id"].unique()))


def get_stop_names():
    return pd.read_csv("zet_processed/tram_stops.csv")[["stop_id", "stop_name"]].set_index("stop_id").T.to_dict()


from collections import defaultdict
'''
goes through stop_times and computes time(s)/location(m) distance between neighbour stops
this is done separately for each service type (from 1 to 9 depending on day type: work day, weekend, holiday...) and each tram
expressed as a list of all values
later used in calculating avg and median time
'''
def calculate_stop_distances_times():
    stop_data = pd.read_csv("zet_processed/tram_stops.csv").set_index("stop_id").T.to_dict("list")

    stop_times_data_grouped = pd.read_csv("zet_processed/tram_stop_times.csv").groupby("trip_id")

    stop_pair_distances = defaultdict(list)

    prev = None
    for name,group in stop_times_data_grouped:
        service_type = int(name.split("_")[1])
        tram_id = int(name.split("_")[3])
        index = -1
        for _,current in group.iterrows():
            index += 1
            if index == 0:
                prev = current
                continue

            prev_stop_id = prev["stop_id"]
            current_stop_id = current["stop_id"]

            if prev_stop_id == current_stop_id:
                continue

            prev_time = prev["time"]
            current_time = current["time"]
            time_diff = calculate_time_diff(prev_time, current_time)

            prev_stop_details = stop_data[prev_stop_id]
            current_stop_details = stop_data[current_stop_id]

            distance = calculate_distance_haversine(prev_stop_details[1], prev_stop_details[2], current_stop_details[1], current_stop_details[2])

            stop_pair_distances[(tram_id, service_type, prev_stop_id, current_stop_id)].append((time_diff, distance))

            prev = current


    with open("temp/stop_distances_list.txt", "w") as file:
        file.write("tram_id,service_type,from_stop,to_stop,distance_m,times_s\n")
        out = [f"{k[0]},{k[1]},{k[2]},{k[3]},{v[0][1]},{';'.join([str(a[0]) for a in v])}\n" for k,v in stop_pair_distances.items()]

        file.writelines(out)

import csv
import statistics
'''
takes output from above method and calculates avg and median times along with standard deviation for travel times between stops
'''
def process_stop_distances():
    data = pd.read_csv("temp/stop_distances_list.txt")
    
    data["arithmetic_mean_time"] = data.apply(lambda row : statistics.mean(list(map(int,row["times_s"].split(";")))), axis=1)
    data["geometric_mean_time"] = data.apply(lambda row : statistics.geometric_mean(list(map(int,row["times_s"].split(";")))), axis=1)
    data["median_time"] = data.apply(lambda row : statistics.median(list(map(int,row["times_s"].split(";")))), axis=1)

    def calculate_stdev(row):
        times = list(map(int,row["times_s"].split(";")))
        if len(times) > 1:
            return statistics.stdev(times)
        else:
            return 0.0

    data["stdev"] = data.apply(calculate_stdev, axis=1)

    data.drop('times_s', axis=1, inplace=True)

    data.to_csv("zet_processed/tram_stop_distances_immediate.csv", index=False)

from dataclasses import make_dataclass
'''
goes through stop_times and finds all unique routes (stop1 -> stop2 -> stop3...)
then removes duplicate neighbour stops (for some reason ZET files have them)
'''
def find_all_routes_between_stops():
    stop_times_grouped_by_trip_id = pd.read_csv("zet_processed/tram_stop_times.csv").groupby("trip_id")
    tram_trips = pd.read_csv("zet_processed/tram_trips.csv").set_index("trip_id").T.to_dict()

    routes = defaultdict(int)
    for k,v in stop_times_grouped_by_trip_id:
        group = stop_times_grouped_by_trip_id.get_group(k)
        trip = tram_trips[k]
        direction_id = trip["direction_id"]
        service_type = int(k.split("_")[1])
        tram_id = int(k.split("_")[3])
        stops_in_route = tuple(group["stop_id"])
        routes[(tram_id, service_type, direction_id, stops_in_route)] += 1

    routes_duplicates_removed = []
    Route = make_dataclass("Route", [("tram_id", int), ("service_type", int), ("direction_id", int), ("route", str), ("frequency", int)])

    for (tram_id,service_type,direction_id,r),n in routes.items():
        r_list = list(r)
        for i,s in enumerate(r_list[:-1]):
            if r[i+1] == s:
                del r_list[i]

        routes_duplicates_removed.append(Route(tram_id, service_type, direction_id, ';'.join(map(str,r_list)), n))

    routes_df = pd.DataFrame(routes_duplicates_removed).sort_values(by=["tram_id", "service_type", "frequency"], ascending=[True, True, False])

    print(routes_df)

    routes_df.to_csv("zet_processed/tram_stop_routes.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    # with open("zet_processed/tram_stop_routes.csv", "w") as file:
    #     file.write("tram_id,service_type,route\n")
    #     for tram_id,service_type,r in routes_duplicates_removed:
    #         file.write(f"{tram_id},{service_type},{';'.join(map(str,r))}\n")

'''
find all routes that are part of another
'''
def find_sub_routes():
    routes = None
    with open("zet_processed/tram_stop_routes.csv") as file:
        routes = [l.strip() for l in file.readlines()]

    dups = set()

    for i in range(len(routes)):
        ith = routes[i]
        for j in range(len(routes)):
            jth = routes[j]
            if i != j and ith in jth:
                print(f"{i+1}th in {j+1}th")
                dups.add(i)

    print(dups, len(dups))
'''
goes through file find_all_routes_between_stops() generated and replaces stop_ids with stop_names
'''
def name_stop_routes():
    stop_names = get_stop_names()

    data = pd.read_csv("zet_processed/tram_stop_routes.csv")

    data["route"] = data.apply(lambda row : ";".join(map(lambda stop : f"{stop_names[int(stop)]['stop_name']}({stop})", row["route"].split(";"))), axis=1)

    data.to_csv("zet_processed/tram_stop_routes_named.csv", index=False)


from graphviz import Digraph
'''
creates .dot file for graphviz from the stop_distances_immediate.csv
'''
def create_dot_file():
    dot = Digraph()

    stop_names = get_stop_names()

    for k,v in stop_names.items():
        dot.node(str(k), v["stop_name"])

    stop_distances = pd.read_csv("zet_processed/stop_distances_immediate.csv")


    for i, row in stop_distances.iterrows():
        dot.edge(str(int(row["from_stop"])), str(int(row["to_stop"])), f"{row['distance_m']:.1f}m, {row['median_time']}s")

    print(dot.source)

'''
extends tram_stop_times.csv by adding a column with stop names for easier readability
'''
def set_stop_names_in_stop_times():
    stop_names = get_stop_names()
    stop_times = pd.read_csv("zet_processed/tram_stop_times.csv")
    stop_times["stop_name"] = stop_times.apply(lambda row:stop_names[row["stop_id"]]["stop_name"], axis=1)
    stop_times = stop_times[["trip_id","time","stop_id","stop_name","stop_sequence","stop_headsign"]]
    stop_times.to_csv("zet_processed/tram_stop_times_with_named_stops.csv", index=False)

'''
helper function
reads existing routes from file and returns them as a list of tuples containing tram_id and list of stop_ids (int)
#NEEDS FIX FOR DIFFERENTIATING SERVICE_TYPES
'''
def load_existing_routes():
    with open("zet_processed/tram_stop_routes.csv") as file:
            return [(l.split(",")[0],list(map(int,l.split(",")[1].split(";")))) for l in file.readlines()[1:]]

'''
returns all routes between 2 stops without duplicates (only routes longer than 2)
#NEEDS FIX FOR DIFFERENTIATING SERVICE_TYPES
'''
def find_all_routes_between_2_stops(stop1_id, stop2_id, existing_routes=load_existing_routes()):
    found_routes = set()
    for j,(tram_id,stops) in enumerate(existing_routes):
            stop1_index = len(stops)
            for i,s in enumerate(stops):
                if s == stop2_id:
                    if stop1_index == len(stops): break
                    else:
                        found_routes.add((tram_id, tuple(stops[stop1_index:i+1])))
                if s == stop1_id: stop1_index = i

    return found_routes



import itertools
from dataclasses import dataclass
from dataclasses import field
#NEEDS FIX FOR DIFFERENTIATING SERVICE_TYPES
@dataclass
class Route:
    tram_id : int
    from_stop : int
    to_stop : int
    stops_in_route : list[int] = field(default_factory=list)


'''
uses the method above to find routes (for example between stop 1 and 5: 1 => 3 => 4 => 2 => 5)
includes trams
NEEDS FIX FOR DIFFERENTIATING SERVICE_TYPES
NEEDS FIX FOR LIST OF STOPS IN ROUTE: from "[209, 208, 264]" to 209;208;264
'''
def find_all_routes_between_all_stops():
    existing_routes=load_existing_routes()
    stops_ids_list = list(pd.read_csv("zet_processed/tram_stops.csv")["stop_id"].astype(int))


    pairs = list(itertools.combinations(stops_ids_list, 2))

    routes = []

    for stop_i,stop_j in pairs:
        routes_ij = find_all_routes_between_2_stops(stop_i, stop_j, existing_routes)
        for route in routes_ij:
            routes.append(Route(int(route[0]), stop_i, stop_j, list(route[1])))

        routes_ji = find_all_routes_between_2_stops(stop_j, stop_i, existing_routes)

        for route in routes_ji:
            routes.append(Route(int(route[0]), stop_j, stop_i, list(route[1])))

    df = pd.DataFrame(routes)
    print(df)

    df.to_csv("zet_processed/tram_routes_between_all_stop_pairs.csv", index=False)

'''
uses immediate distances to calculate extended ones -> meaning distances between stops that are on the same route but not next to each other
does it as sum of distances in between
different for each tram
NEEDS FIX FOR DIFFERENTIATING SERVICE_TYPES
'''
def calculate_distances_extended():
    stop_distances_immediate = pd.read_csv("zet_processed/tram_stop_distances_immediate.csv").set_index(["tram_id","from_stop","to_stop"]).T.to_dict()
    all_routes = pd.read_csv("zet_processed/tram_routes_between_all_stop_pairs.csv")


    def sum_by_column(item):
        distance_sum = arithmetic_mean_time_sum = geometric_mean_time_sum = median_time_sum = 0
        stops = list(map(int,item["stops_in_route"][1:-1].split(",")))
        tram_id = item["tram_id"]

        for i,stop in enumerate(stops[1:]):
            prev_stop = stops[i]
            all_distances = stop_distances_immediate[(tram_id,prev_stop,stop)]

            distance_sum += all_distances["distance_m"]
            arithmetic_mean_time_sum += all_distances["arithmetic_mean_time"]
            geometric_mean_time_sum += all_distances["geometric_mean_time"]
            median_time_sum += all_distances["median_time"]

        return distance_sum, arithmetic_mean_time_sum, geometric_mean_time_sum, median_time_sum

    distances = all_routes.apply(sum_by_column,result_type="expand",axis=1)

    distances.columns = ["distance_sum_m", "arithmetic_mean_time_sum", "geometric_mean_time_sum", "median_time_sum"]

    all_routes_with_distances = pd.concat([all_routes,distances], axis=1)

    all_routes_with_distances.drop('stops_in_route', axis=1, inplace=True)

    all_routes_with_distances.to_csv("zet_processed/tram_stop_distances_extended.csv", index=False)

    print(all_routes_with_distances)

'''
function filters out all necessary data (including stops, stop_times, distances) so only those that are relevant to these trams remain
aim is to simplify data needed for the simulation
parametar trams is a list of ints representing tram_ids

parameter dir is dir where all the filtered files will be put
parameter service_types is a list of service types (ints 1-9) that'll remain, default is only 1
'''
def filter_for_simulation(trams, service_types=None, dir="filtered/"):
    if service_types is None:
        service_types = [1]
    
    # reading files to filter
    tram_routes = pd.read_csv("zet_processed/tram_routes.csv")
    tram_stop_distances_immediate = pd.read_csv("zet_processed/tram_stop_distances_immediate.csv")
    tram_stop_times = pd.read_csv("zet_processed/tram_stop_times.csv")
    

'''
find the most frequent route for each tram_id for work day (service type 1)
route is expressed as a list of stops
each tram has 2, for each direction
also removes stops that don't appear in most common routes from stop_list
finally it adds a column that says which direction of the stop is viable for spawning
'''
def prepare_simulation_files():
    tram_stop_routes = pd.read_csv("zet_processed/tram_stop_routes.csv")

    tram_stop_routes = tram_stop_routes[tram_stop_routes["tram_id"] < 30]
    tram_stop_routes = tram_stop_routes[tram_stop_routes["service_type"] == 1]
    tram_stop_routes.drop(["service_type"], axis=1, inplace=True)

    tram_stop_routes = tram_stop_routes.sort_values('frequency').drop_duplicates(["tram_id","direction_id"], keep='last').sort_values(by=["tram_id", "direction_id"])
    tram_stop_routes.drop(["frequency"], axis=1, inplace=True)

    # viable for spawning passengers, for example some stops are only one directional and some are last in their route so passengers shouldn't spawn
    viable_stops = set()
    viable_stop_directions = defaultdict(set)
    last_stops = {}

    for _, item in tram_stop_routes.iterrows():
        route = item["route"]
        direction = item["direction_id"]
        stops_in_route = route.split(";")
        for i,stop in enumerate(stops_in_route):
            viable_stops.add(int(stop))
            if i != len(stops_in_route) - 1: viable_stop_directions[int(stop)].add(direction)


    tram_stops = pd.read_csv("zet_processed/tram_stops.csv")
    tram_stops = tram_stops[tram_stops["stop_id"].isin(list(viable_stops))]

    def add_viable_directions(row):
        vsd = viable_stop_directions[row["stop_id"]]

        if len(vsd) == 2: return 2
        elif len(vsd) == 0: return -1
        else: return vsd.pop()

    tram_stops["viable_direction"] = tram_stops.apply(add_viable_directions, axis=1)

    tram_stop_routes.to_csv("simulation/tram_common_routes.csv", index=False)
    tram_stops.to_csv("simulation/tram_viable_stops.csv", index=False)




if __name__ == "__main__":
    prepare_simulation_files()
    



