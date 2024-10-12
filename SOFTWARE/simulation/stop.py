import typing
import pandas as pd
import random
from sklearn.neighbors import BallTree
import numpy as np
from collections import defaultdict
import time

import tram
import passenger
from common import PassengerHolder, Location, PassengerGoal, Logger, interpolate_coordinates, calculate_distance_haversine


Route = list[int]
TramRoute = tuple[int, Route]
TramRoutes = list[TramRoute]
StopDirection = tuple[int,int]
ClosestStops = tuple[np.ndarray, pd.core.frame.DataFrame]

StopRoutes = dict[StopDirection, TramRoutes]

Points = list[Location]
PointRoutes = list[Points]

'''
computes possible next stops for a certain stop/direction pair
returns a dictionary that maps (stop_id, direction_id) -> list of (tram_id, next_stops)
'''
def connect_stops_routes() -> StopRoutes:
    routes = pd.read_csv("tram_common_routes.csv")

    stop_routes = defaultdict(list)

    for _, item in routes.iterrows():
        tram_id = item["tram_id"]
        direction_id = item["direction_id"]
        route = list(map(int, item["route"].split(";")))

        for i, stop in enumerate(route[:-1]):
            stop_routes[(stop, direction_id)].append((tram_id, route[i+1:]))

    return stop_routes

'''
connects stops and routes in a way that for each stop it lists indexes of routes that pass that stop
saves results to a file
'''
def connect_possible_routes_for_stops():
    routes = pd.read_csv("tram_common_routes.csv")

    possible_stop_routes = defaultdict(list)

    for i, item in routes.iterrows():
        route = list(map(int, item["route"].split(";")))

        for stop in route:
            possible_stop_routes[stop].append(i)

    with open("routes_per_stop.csv", "w") as file:
        file.write(f"stop_id,route_indexes\n")
        for k,v in possible_stop_routes.items():
            file.write(f"{k},{';'.join(map(str,v))}\n")



class StopDetails:
    def __init__(self):
        self.stops = pd.read_csv("tram_viable_stops.csv")
        self.stops_dict = self.stops.set_index("stop_id").T.to_dict()

        self.bt = BallTree(np.deg2rad(self.stops[['stop_lat', 'stop_lon']].values), metric='haversine')

    def get_stop_details(self, stop_id):
        return self.stops_dict[stop_id]

    def get_all_stops(self):
        return self.stops_dict

    def get_closest_stop(self, location: Location, how_many: int = 1) -> ClosestStops:
        query_lats = location[0]
        query_lons = location[1]

        distances, indices = self.bt.query(np.deg2rad(np.c_[query_lats, query_lons]), k=how_many)

        return (distances[0] * 6371000, self.stops.iloc[indices[0]])

def calculate_direction(position1: Location, position2: Location) -> float:
    diff = np.array([position2[0] - position1[0], position2[1] - position1[1]])

    return np.arctan2(*diff)

def get_interpolated_stop_routes(interpolation_distance: int = 50) -> pd.core.frame.DataFrame:
    stopDetails = StopDetails()
    routes = pd.read_csv("tram_common_routes.csv")

    stop_distances = pd.read_csv("tram_stop_distances_immediate.csv")
    stop_distances.drop(["tram_id","service_type","arithmetic_mean_time","geometric_mean_time","median_time","stdev"], axis=1, inplace=True)
    stop_distances = stop_distances.drop_duplicates()
    stop_distances = stop_distances.set_index(["from_stop", "to_stop"]).T.to_dict("dict")

    def interpolate_route(row):
        route = list(map(int,row["route"].split(";")))
        total_distance = sum([stop_distances[(s1,s2)]["distance_m"] for s1,s2 in zip(route[:-1], route[1:])])
        
        # stores positions for original stops in route based on distance
        # for example if route has stops 1,2,3 with distances 200 (1,2) and 500 (2,3) and we want to interpolate every 100m
        # total distance is 700 and stop positions will be [0,2,7]
        stop_positions = [0]

        total_interpolation_points = int(total_distance / interpolation_distance)

        stop_locations = [(stopDetails.get_stop_details(stop)["stop_lat"], stopDetails.get_stop_details(stop)["stop_lon"]) for stop in route]

        if total_interpolation_points <= len(route): return stop_locations 

        distance_passed = 0

        for s1,s2 in zip(route[:-2], route[1:-1]):
            distance = stop_distances[(s1,s2)]["distance_m"]
            distance_passed += distance

            stop_positions.append(int(total_interpolation_points*distance_passed/total_distance))

        stop_positions.append(total_interpolation_points-1)

        interpolated_points = interpolate_coordinates(stop_locations, stop_positions, total_interpolation_points)

        return interpolated_points

    def direction(row):
        ir = row["interpolated_route"]
        
        first_point = ir[0]
        last_point = ir[-1]

        return calculate_direction(first_point, last_point)

    routes["interpolated_route"] = routes.apply(interpolate_route, axis=1)
    routes["direction"] = routes.apply(direction, axis=1)

    return routes



def save_interpolated_stop_routes(interpolation_distance: int = 50, file_name:str = "interpolated_tram_common_routes.csv", ):
    routes = get_interpolated_stop_routes(interpolation_distance)

    routes["interpolated_route"] = routes.apply(lambda row : ";".join([f"{ip[0]},{ip[1]}" for ip in row["interpolated_route"]]), axis=1)

    routes.to_csv("interpolated_tram_common_routes.csv", index=False)


def get_saved_interpolated_stop_routes(file_name:str = "interpolated_tram_common_routes.csv") -> pd.core.frame.DataFrame:
    interpolated_routes = pd.read_csv(file_name)

    def parse(row):
        return [(float(ip.split(",")[0]), float(ip.split(",")[1])) for ip in row["interpolated_route"].split(";")]

    interpolated_routes["interpolated_route"] = interpolated_routes.apply(parse, axis=1)

    return interpolated_routes


class RouteStopComparator:
    def __init__(self, interpolation_distance: int):
        self.interpolated_routes = get_interpolated_stop_routes(interpolation_distance)
        self.stopDetails = StopDetails()
        self._prepare_possible_tram_routes()
        
    def _prepare_possible_tram_routes(self):
        rps = pd.read_csv("routes_per_stop.csv")
        rps["route_indexes"] = rps.apply(lambda row : list(map(int,row["route_indexes"].split(";"))), axis=1)
        rps = rps.set_index("stop_id").T.to_dict()

        self.routes_per_stop = rps


    # based on closest stops for first and last point, it returns list of possible tram routes (ones that pass those stops)
    def get_possible_tram_routes(self, points: Points):
        possible_tram_routes = set()

        first = points[0]
        last = points[-1]
        distances, closest_stops = self.stopDetails.get_closest_stop(first, 3)
        distances2, closest_stops2 = self.stopDetails.get_closest_stop(last, 3)

        stops = pd.concat([closest_stops, closest_stops2])
        distances = np.concatenate([distances,distances2])
        
        for i,distance in enumerate(distances):
            if distance < 380:
                stop = stops.iloc[i]
                stop_id = stop["stop_id"]
                possible_tram_routes.update(self.routes_per_stop[stop_id]["route_indexes"])

        return possible_tram_routes


    def find_closest_tram_route(self, points: Points, accuracy: int = 100):
        # print("finding closest tram route for", ";".join(map(lambda x : f"{x[0]},{x[1]}",points)))
        points_direction = calculate_direction(points[0], points[-1])

        possible_tram_routes = self.get_possible_tram_routes(points)
        # print(possible_tram_routes)

        filtered_interpolated_routes = self.interpolated_routes

        # filtered_interpolated_routes = filtered_interpolated_routes[abs(self.interpolated_routes["direction"] - points_direction) <= 2.4 ]
        # print(filtered_interpolated_routes)
        filtered_interpolated_routes = filtered_interpolated_routes[filtered_interpolated_routes.index.isin(possible_tram_routes) ]


        filtered_interpolated_routes_list = list(filtered_interpolated_routes["interpolated_route"])

        (index, similarity) = self.find_closest_route(points, filtered_interpolated_routes_list)

        res = filtered_interpolated_routes.iloc[[index]]
        # print("yoo",res["tram_id"].values[0])

        return ((res["tram_id"].values[0], res["direction_id"].values[0]), similarity)



    def find_closest_route(self, points: Points, routes: PointRoutes, radius: float = 150):
        radius /= 6371000

        bt = BallTree(np.deg2rad(points), metric='haversine')

        res = [np.count_nonzero(bt.query_radius(np.deg2rad(r), radius, count_only=True)) for r in routes]
        
        if len(res) > 0:
            argmax = np.argmax(res)
            similarity = res[argmax] / len(routes[argmax])
            return (argmax, similarity)
        else:
            print("yoooooo", points, routes, res)
            return (0,0)

       

        bts = [BallTree(np.deg2rad(r), metric='haversine') for r in routes]
        points = np.deg2rad(points)
        for i,bt in enumerate(bts):
            res = bt.query_radius(points, radius, count_only=True)
            # print(i, res, f"{np.count_nonzero(res) / len(res)}")

        res = [np.count_nonzero(bt.query_radius(points, radius, count_only=True)) for bt in bts]
        argmax = np.argmax(res)
        similarity = res[argmax] / len(routes[argmax])

        return (argmax, similarity)


    def get_route_similarity(self, points1: Points, points2: Points, radius: float = 125):
        radius /= 6371000
        points2 = np.deg2rad(points2)

        bt = BallTree(np.deg2rad(points1), metric='haversine')

        in_radius = bt.query_radius(points2, radius, count_only=True)

        return np.count_nonzero(in_radius) / len(in_radius)

    def detect_intruders(self, points: Points) -> list[int]:
        # print("detecting intruders", points)
        
        avg = np.average(points, axis=1)

    @staticmethod
    def detect_outlier_by_avg(points, max_distance=40):
        # print("detecting outliers by avg")
        center = average_point(points)
        # print(f"{center=}")

        distances = [calculate_distance_haversine(center, p) for p in points]
        # print(f"{distances=}")

        avg_distance = sum(distances) / len(distances)
        # print(f"{avg_distance=}")

        distance_diffs = [(i, d - avg_distance) for i,d in enumerate(distances) if (d - avg_distance) > max_distance]
        # print(f"{distance_diffs=}")
        if len(distance_diffs) == 0: return None
        possible_outlier = max(distance_diffs, key = lambda p : p[1])
        # print(f"{possible_outlier=}")

        return possible_outlier[0]

    @staticmethod
    def detect_outlier_by_distances(points, max_distance=40):
        # print("detecting outliers by distances")

        pair_distances = defaultdict(float)
        avg_distances = defaultdict(float)

        for i in range(len(points)):
            pi = points[i]
            for j in range(i+1, len(points)):
                pj = points[j]

                distance = calculate_distance_haversine(pi,pj)
                pair_distances[(i,j)] = distance
                avg_distances[i] += distance / (len(points) -1)
                avg_distances[j] += distance / (len(points) -1)

        # print(f"{pair_distances=}")
        # print(f"{avg_distances=}")

        avg_avg = sum(avg_distances.values()) / len(points)
        # print(f"{avg_avg=}")

        distance_diffs = [(i, d - avg_avg) for i,d in enumerate(avg_distances.values()) if (d - avg_avg) > max_distance]
        # print(f"{distance_diffs=}")
        if len(distance_diffs) == 0: return None

        possible_outlier = max(distance_diffs, key = lambda p : p[1])
        # print(f"{possible_outlier=}")

        return possible_outlier[0]


def average_point(points: Points) -> Location:
    avg = np.average(points, axis=0)
    return (*avg,)

def remove_outliers(points: Points, method):
    outlier = method(points)
    while outlier is not None:
        del points[outlier]
        outlier = method(points)

    return points


class StopObject(PassengerHolder):
    parked_tram: 'tram.Tram' = None
    passed_time: int = 0
    holder_type: str = "stop"

    already_notified: set[int] = set()

    logger: Logger

    print_enabled: bool = True

    def __init__(self, lat: float, lon: float, stop_id: int, name: str, passenger_manager: 'passenger.PassengerManager', direction_id: int, logger: Logger, print_enabled: bool = True):
        self.lat = lat
        self.lon = lon
        self.stop_id = stop_id
        self.name = name
        self.passenger_manager = passenger_manager
        self.direction_id = direction_id
        self.print_enabled = print_enabled
        self.logger = logger

        self.passengers = set()

    @property
    def holder_id(self):
        return self.stop_id

    @property
    def holder_direction(self):
        return self.direction_id

    def __repr__(self) -> str:
        return f"Stop {self.name} ({self.stop_id}) ({self.direction_id})"

    def _log(self, message: str, message_type: str = "default"):
        self.logger.log_message(f"{self}: {message}", message_type)

    @classmethod
    def from_stop_details_dict(cls, stop_id: int, stop_details: dict, passenger_manager: 'passenger.PassengerManager', direction_id: int, logger: Logger, print_enabled: bool = True):
        return cls(lat = stop_details["stop_lat"], lon = stop_details["stop_lon"], stop_id = stop_id, name = stop_details["stop_name"], passenger_manager = passenger_manager, direction_id = direction_id, logger = logger ,print_enabled = print_enabled)

    def get_location(self) -> Location:
        return (self.lat, self.lon)

    def add_passenger(self, passenger_id : int) -> bool:
        assert passenger_id not in self.passengers, f"Passenger {passenger_id} already in!"

        self.passengers.add(passenger_id)
        if self.print_enabled: print(f"{self} added a passenger {passenger_id}", self.passengers)
        return True

    def remove_passenger(self, passenger_id: int):
        assert passenger_id in self.passengers, f"Can't remove passenger {passenger_id}, not in!"
        
        self.passengers.remove(passenger_id)
        if self.print_enabled: print(f"{self} removed passenger {passenger_id}")

    def tram_arrival(self, tram: 'tram.Tram'):
        if self.print_enabled: print(f"{tram} arrived at stop {self.stop_id}, {self.direction_id}")
        self.parked_tram = tram
        self._log(f"Tram {tram} arrived", "TRAM ARRIVAL/DEPARTURE")
        # self.already_notified = set()

    def tram_departure(self, tram: 'tram.Tram'):
        if tram == self.parked_tram:
            if self.print_enabled:  print(f"tram {tram.tram_index} departing from stop {self.stop_id}, {self.direction_id}")
            self.parked_tram = None
            self._log(f"Tram {tram} departed", "TRAM ARRIVAL/DEPARTURE")

            


    def next_step(self):
        to_notify = self.passengers.copy() # - self.already_notified
        if self.print_enabled and len(to_notify) > 0: print(f"{self} about to notify passengers {to_notify} that {self.parked_tram} is parked")
        if self.parked_tram is not None and len(to_notify) > 0:
            if self.print_enabled: print(f"notifying passengers {to_notify} that tram {self.parked_tram} is at STOP {self}")
            self.passenger_manager.notify_tram_arrival(to_notify, self.parked_tram)
            # self.already_notified = self.already_notified | self.passengers

        self.passed_time += 1

class StopManager:
    passed_time: int = 0
    stops : dict[(int, int), StopObject] = {}
    total_spawned: int = 0
    stop_routes: StopRoutes = {}
    print_enabled: bool = True
    spawn_viable_stops: set[tuple[int,int]] = set()
    max_passenger_num: int = 100
    logger: Logger


    def __init__(self, stopDetails: StopDetails, passenger_manager: 'passenger.PassengerManager', logger: Logger, print_enabled: bool = True, max_passenger_num: int = 100):
        self.passengerManager = passenger_manager
        self.print_enabled = print_enabled
        self.logger = logger
        self.max_passenger_num = max_passenger_num
        self.stop_routes = connect_stops_routes()

        # NEEDS TO BE LOOKED BECAUSE THIS MIGHT BE CREATING NON EXISTING STOPS (ONES WITH ONLY ONE DIRECTION)
        for stop_id, stop_details in stopDetails.get_all_stops().items():
            self.stops[(stop_id, 0)] = StopObject.from_stop_details_dict(stop_id = stop_id, stop_details = stop_details, passenger_manager = passenger_manager, direction_id = 0, logger = logger, print_enabled=self.print_enabled)
            self.stops[(stop_id, 1)] = StopObject.from_stop_details_dict(stop_id = stop_id, stop_details = stop_details, passenger_manager = passenger_manager, direction_id = 1, logger = logger, print_enabled=self.print_enabled)

            viable_direction = stop_details["viable_direction"]
            
            if viable_direction == 2:
                self.spawn_viable_stops.add((stop_id,0))
                self.spawn_viable_stops.add((stop_id,1))
            elif viable_direction != -1:
                self.spawn_viable_stops.add((stop_id, viable_direction))


    def _get_random_goal(self, stop_id: int, direction_id: int) -> PassengerGoal:
        tram_routes = self.stop_routes[(stop_id, direction_id)]

        if self.print_enabled: print(f"about to return random goal from {tram_routes} for {stop_id}, {direction_id}")

        tram_goal, route_goal = random.choice(tram_routes)
        stop_goal = random.choice(route_goal)

        return (tram_goal, stop_goal)


    def get_stop(self, stop_id: int, direction_id: int):
        return self.stops[(stop_id, direction_id)]

    def add_passenger(self, stop_id: int, passenger_id: int):
        assert stop_id in self.stops, f"StopManager: adding passenger error, Stop {stop_id} doesn't exist"

        self.stops[stop_id].add_passenger(passenger_id)

    def remove_passenger(self, stop_id: int, passenger_id: int):
        assert stop_id in self.stops, f"StopManager: removing passenger error, Stop {stop_id} doesn't exist"
        
        self.stops[stop_id].remove_passenger(passenger_id)

    def tram_arrival(self, stop_id: int, tram: 'tram.Tram'):
        self.stops[(stop_id, tram.direction_id)].tram_arrival(tram)

    def tram_departure(self, stop_id: int, tram: 'tram.Tram'):
        self.stops[(stop_id, tram.direction_id)].tram_departure(tram)

    def set_custom_spawn_viable_stops(self, spawn_viable_stops):
        self.spawn_viable_stops = spawn_viable_stops

    def spawn_passengers(self):
        if self.print_enabled: print("trying to spawn some passangers")

        num_alive = self.passengerManager.num_alive
        diff_alive = num_alive - self.max_passenger_num

        if self.max_passenger_num == 0 or (diff_alive > 0 and random.random() <= 0.7 + diff_alive / 100):
            if self.print_enabled: print(f"too many passengers {num_alive}, skiping")
            return


        chosen_stops = random.sample(self.spawn_viable_stops, k=5)
        print("choesn stops", chosen_stops)
        # chosen_stops = [(103,0),(169,0),(105,0),(106,0),(296,0),(299,0),(163,0)]
        for stop_id, direction_id in chosen_stops:
            to_spawn = random.randint(-15,15)
            for _ in range(to_spawn):
                random_goal = self._get_random_goal(stop_id, direction_id)
                self.spawn_passenger(random_goal, (stop_id, direction_id))

                
    def spawn_passenger(self, goal, stop_details):
        stop = self.stops[stop_details]

        self.total_spawned += 1
        p = self.passengerManager.create_new_passenger(goal)

        self.passengerManager.bind_passenger(p, stop)
        stop.add_passenger(p)

        if self.print_enabled: print(f"spawned new passenger {p} at {stop} with goals {goal}")

    def next_step(self):
        if self.passed_time % 5 == 0: self.spawn_passengers()

        for s in self.stops.values():
            s.next_step()

        self.passed_time += 1

    def enable_print(self, print_enabled:bool):
        self.print_enabled = print_enabled
        for s in self.stops.values(): s.print_enabled = print_enabled

def main():
    sd = StopDetails()

    customRoutes = [
        ('''
    45.78168651410924,16.00320344510077
    45.78420048609953,16.00440507473944
    45.78647493450425,16.005263381624207
    45.788928840549296,16.005263381624207
    45.79054476846264,16.00423341336249
    45.792818958011765,16.002860122346863
    45.79425524044109,16.001572662019715
    45.7961702260999,16.00045686306952
    45.79868354491155,15.99865441861151
    45.801495933936714,15.997023635530455
    45.80442784794348,15.994362884187682
    45.80550483882798,15.991530471467955
    45.80819722494651,15.991702132844908
    ''', (7,0)),
        ('''
    45.78699823687892,15.95469571775069
    45.78753690337865,15.955296532570026
    45.78897332192935,15.956240670143268
    45.789930913731986,15.956841484962604
    45.79142711798838,15.957871453224323
    45.7926240524747,15.958386437355182
    45.794838313508315,15.95967389768233
    45.795616276229005,15.960360543190143
    45.79729185135556,15.962162987648151
    45.798309140244456,15.962592141090534
    45.79896737609948,15.965081231056354
    45.79944608820097,15.967398659645221
    45.79908705451046,15.969544426857135
    45.79926382570096,15.972039963642514
    45.799622858252434,15.973477627674496
    45.79933862433997,15.977061058918393
    45.80028107858821,15.979271199146664
    45.79963781789186,15.981438424030697
    45.80035587586161,15.983519818226254
    45.79982460662897,15.986653822674052
    45.8010811977905,15.988885420574443
    45.800213554540704,15.992232817425029
    ''', (5,0)),
        ('''
    45.77584518958308,15.969997565913694
    45.775007085933375,15.962787788081663
    45.777162184155216,15.956264655757444
    45.778718592184696,15.952488105464475
    45.78302841846399,15.95162979857971
    45.786978799893305,15.953003089595335
    45.78949253323356,15.955234687495725
    45.7924849251172,15.95712296264221
    45.79463934779307,15.961757819819944
    45.797511781805355,15.961242835689085
    45.80122178993281,15.963474433589475
    45.804213552033936,15.96536270873596
    ''', (14,1)),
    ]

    rts = RouteStopComparator(100)

    start_time = time.time()
    for i,(route_str, correct) in enumerate(customRoutes):
        customRoute = [(float(l.split(",")[0]), float(l.split(",")[1])) for l in route_str.strip().split("\n")]

        res = rts.find_closest_tram_route(customRoute)
        print(f"route {i}:\n\tresult: {res}\n\tcorrect: {correct}")

    print(f"elapsed time: {time.time() - start_time}")

def main2():
    r1 = [(45.8052801331561, 15.973910335306636),(45.805567088976865, 15.974021955773658),(45.80549692399943, 15.97416833297299),(45.80552811523457, 15.974678209609971),(45.805209681860674, 15.97469335329969),(45.8056356454234, 15.975336097433383),(45.80543029458681, 15.975585413622898),(45.80536803353358, 15.976154265815238),(45.80552034619163, 15.9760298622195),(45.805317354942346, 15.976470606972752)]

    r2 = [(45.80591665367689, 15.974261579693575),(45.80510371094165, 15.974372411917031),(45.80527250613577, 15.973722858735938),(45.80537496380425, 15.974602044505431),(45.805073292308805, 15.974934687790343),(45.80582263042133, 15.975008099529822),(45.805588961945205, 15.975258633141033),(45.8052051214538, 15.97550949854842),(45.804940466663155, 15.976012586493654),(45.80550673980835, 15.97686068788748)]

    r3 = [(45.80511307512138,15.971987105715199),(45.805487028049534,15.97246990333788),(45.80516542868245,15.973306752550526),(45.805389800529696,15.973628617632313),(45.80509811695206,15.974283076631947),(45.80533744717951,15.97469077240221),(45.805060721511126,15.97519502769701),(45.80541971670772,15.975838757860584)]

    a = RouteStopComparator(100)

    # print(a.get_route_similarity(r1,r3,50))
    print(a.find_closest_tram_route(r1))

def main3():
    points = '''45.80291703397418,15.986997538232787
45.8025430637945,15.986632757806762
45.80248322833281,15.987297945642455
45.80523559305886,15.98693316521643'''
    points = [tuple(map(float, p.strip().split(","))) for p in points.split("\n")]
    
    def method(points): return RouteStopComparator.detect_outlier_by_avg(points, 40)
    def method2(points): return RouteStopComparator.detect_outlier_by_distances(points, 40)

    print("about to remove outliers from:")
    for p1,p2 in points:
        print(f"{p1},{p2}")

    new_points = remove_outliers(points, method2)
    print("removed outliers")
    for p1,p2 in new_points:
        print(f"{p1},{p2}")





if __name__ == "__main__":
    main3()
