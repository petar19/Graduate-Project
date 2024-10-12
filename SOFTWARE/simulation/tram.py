import pandas as pd
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import random

import stop
import passenger

from common import PassengerHolder, Location, Logger


class TramState(Enum):
    DRIVING, JAM, STOP, RESTING = range(4)


@dataclass()
class Route:
    tram_id: int
    start_stop: int
    final_stop: int
    stops: list[int] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)

def combine_routes_and_distances() -> tuple[Route]:
    routes = pd.read_csv("tram_common_routes.csv")
    distances = pd.read_csv("stop_distances.csv")

    routes["stops"] = routes.apply(lambda row : list(map(int,row["route"].split(";"))), axis=1)
    routes.drop("route", axis=1, inplace=True)
    routes["start_stop"] = routes.apply(lambda row : row["stops"][0], axis=1)
    routes["final_stop"] = routes.apply(lambda row : row["stops"][-1], axis=1)

    distances = distances[distances["service_type"] == 1]
    distances = distances[distances["tram_id"] < 30]
    distances["time"] = distances["median_time"]
    distances.drop(["service_type", "arithmetic_mean_time", "geometric_mean_time", "median_time", "stdev"], axis=1, inplace=True)
    distances = distances.set_index(["tram_id","from_stop","to_stop"]).T.to_dict()

    routes_list = []
    for _, group in routes.groupby(["tram_id"]):
        route_pair = []
        for _,route in group.iterrows():
            tram_id = route["tram_id"]
            distances_between_stops = []
            times_between_stops = []
            speeds_between_stops = []
            route_stops = route["stops"]
            
            for i,next_stop in enumerate(route_stops[1:]):
                current_stop = route_stops[i]
                d = distances[(tram_id, current_stop, next_stop)]
                distance_m = d["distance_m"]
                time = d["time"]
                speed = 1 / time
                distances_between_stops.append(distance_m)
                times_between_stops.append(time)
                speeds_between_stops.append(speed)
            
            r = Route(tram_id, route["start_stop"], route["final_stop"], route_stops, distances_between_stops, times_between_stops, speeds_between_stops)
            route_pair.append(r)

        routes_list.append(tuple(route_pair))

    return routes_list


class QueueManager:
    def __init__(self):
        self._route_part_queue = defaultdict(list)
        self.currently_jammed = {}

    def jam(self, route_part, tram_index):
        return 0


    def get_next_in_line(self, route_part, tram_index):
        assert route_part in self._route_part_queue, f"route_part {route_part} doesn't exist"

        trams_on_route_part = self._route_part_queue[route_part]

        n = len(trams_on_route_part)

        for i, (index, _) in enumerate(trams_on_route_part):
            if tram_index == index and i != n - 1: return trams_on_route_part[i + 1][1]

        return -1.0

    def update(self, route_part, tram_index, route_percentage):
        trams_on_route_part = self._route_part_queue[route_part]

        target = None
        for i, (index, _) in enumerate(trams_on_route_part):
            if tram_index == index: target = i

        if target is not None:
            trams_on_route_part[target] = (tram_index, route_percentage)
        else:
            trams_on_route_part.insert(0, (tram_index, route_percentage))



    def sign_out(self, route_part, tram_index):
        assert route_part in self._route_part_queue, f"route_part {route_part} doesn't exist"

        trams_on_route_part = self._route_part_queue[route_part]

        target = None
        for i, (index, _) in enumerate(trams_on_route_part):
            if tram_index == index: target = i

        if target is not None:
            del trams_on_route_part[target]

        if len(trams_on_route_part) == 0:
            del self._route_part_queue[route_part]
    
@dataclass()
class Tram(PassengerHolder):
    tram_index: int
    direction_id: int
    speed_modifier: float
    stop_details: 'stop.StopDetails'
    queueManager: QueueManager
    passengerManager: 'passenger.PassengerManager'
    stopManager: 'stop.StopManager'
    logger: Logger
    route: tuple[Route] = field(default_factory=tuple)
    current_route: Route = field(init=False)
    current_stop_id: int = field(init=False)
    next_stop_id: int = field(init=False)
    current_stop: any = field(init=False)
    next_stop: any = field(init=False)
    current_direction_vector: tuple[float] = field(init=False)
    current_speed: float = field(init=False)
    lat: float = field(init=False)
    lon: float = field(init=False)
    tram_number: int = field(init=False)
    headsign: str = field(init=False)
    current_stop_index: int = 0
    distance_travelled_percentage: float = 0.0
    state: TramState = TramState.DRIVING
    current_wait_time: float = 0.0
    jam_waiting_time: int = 10
    stop_waiting_time: int = 30
    rest_waiting_time: int = 90
    passenger_limit: int = 100
    passengers: set[int] = field(default_factory=set)
    print_enabled: bool = True
    holder_type: str = "tram"


    @property
    def holder_id(self):
        return self.tram_number

    @property
    def holder_direction(self):
        return self.direction_id


    def __repr__(self) -> str:
        return f"Tram {self.tram_number} ({self.tram_index}) ({self.direction_id})"

    def _print(self, message):
        if self.print_enabled: print(f"{self}: {message}")

    def _log(self, message, message_type="default"):
        self.logger.log_message(f"{self}:{message}", message_type)

    def get_location(self):
        return (self.lat, self.lon)

    def add_passenger(self, passenger_id : int) -> bool:
        assert passenger_id not in self.passengers, f"Passenger {passenger_id} already in!"

        if len(self.passengers) >= self.passenger_limit: return False
        else:
            self.passengers.add(passenger_id)
            return True

    def remove_passenger(self, passenger_id):
        assert passenger_id in self.passengers, f"Can't remove passenger {passenger_id}, not in!"

        
        self.passengers.remove(passenger_id)
        if self.print_enabled: print(f"removed passenger {passenger_id} from {self}")


    def _enter_stop_state(self):
        if self.state == TramState.RESTING:
            self._print(f"no longer RESTING, now STOPing")
        else:
            self._print(f"STOPed waiting for passengers")

        self.state = TramState.STOP
        self.current_wait_time = self.stop_waiting_time

        self.stopManager.tram_arrival(self.next_stop_id, self)

    def _enter_rest_state(self):
        self._print(f"finished the route, now RESTING")
        self.stopManager.tram_arrival(self.next_stop_id, self)

        self.direction_id = 1 - self.direction_id
        self.current_route = self.route[self.direction_id]
        self.current_stop_index = 0
        self.state = TramState.RESTING
        self.current_wait_time = self.rest_waiting_time

    def _set_current_info(self):
        self.current_stop_id = self.current_route.stops[self.current_stop_index]
        self.next_stop_id =self.current_route.stops[self.current_stop_index + 1]
        self.current_stop = self.stop_details.get_stop_details(self.current_stop_id)
        self.next_stop = self.stop_details.get_stop_details(self.next_stop_id)
        self.current_direction_vector = (self.next_stop["stop_lat"] - self.current_stop["stop_lat"], self.next_stop["stop_lon"] - self.current_stop["stop_lon"])
        self.current_speed = self.current_route.speeds[self.current_stop_index] * self.speed_modifier
        self.queueManager.update((self.current_stop_id, self.next_stop_id), self.tram_index, 0.0)

    def __post_init__(self):
        self.current_route = self.route[self.direction_id]
        first_stop_details = self.stop_details.get_stop_details(self.current_route.start_stop)
        final_stop_details = self.stop_details.get_stop_details(self.current_route.final_stop)
        self.lat = first_stop_details["stop_lat"]
        self.lon = first_stop_details["stop_lon"]
        self.tram_number = self.current_route.tram_id
        self.headsign = final_stop_details["stop_name"]

        self._set_current_info()

        self._print(f"created! Heading for {self.headsign}\ncurrent location: {self.lat},{self.lon}")
        self._log(f"Created! Heading for {self.headsign}", "TRAM")


    def next_step(self, time):
        if self.state != TramState.DRIVING:
            time = self.wait(time)

            if self.state == TramState.JAM and time > 0:
                self.state = TramState.DRIVING
                self._print(f"no longer JAMed, now DRIVING {time}")
                self.next_step(time)
            elif self.state == TramState.STOP:
                if time > 0:
                    self.state = TramState.DRIVING
                    self._print(f"no longer STOPed, now DRIVING {time}")

                    self.queueManager.sign_out((self.current_stop_id, self.next_stop_id), self.tram_index)
                    self.stopManager.tram_departure(self.current_stop_id, self)


                    self._set_current_info()

                    self.next_step(time)
            else:
                if time > 0:
                    self._enter_stop_state()
                    self.next_step(time)
        else:
            jam = self.queueManager.jam((self.current_stop_id, self.next_stop_id), self.tram_index)
            if jam > 0:
                self.state = TramState.JAM
                self.current_wait_time = jam
                self._print(f"unlucky, got JAMed")
                self._log(f"Unlucky, got JAMed", "TRAM")
                self.next_step(time)
            else:
                self.move(time)

    def wait(self, time):
        self.current_wait_time -= time
        return (-1)*self.current_wait_time

    def move(self, time):
        tram_in_front_progress = self.queueManager.get_next_in_line((self.current_stop_id, self.next_stop_id), self.tram_index)

        if tram_in_front_progress > 0 and (tram_in_front_progress - self.distance_travelled_percentage) <= 0.005:
            self._print(f"can't move, blocked by tram in front")
            return

        if tram_in_front_progress > 0:
            self.distance_travelled_percentage = min(tram_in_front_progress - 0.005, self.distance_travelled_percentage + self.current_speed * time)
        else:
            self.distance_travelled_percentage += self.current_speed * time

        self.queueManager.update((self.current_stop_id, self.next_stop_id), self.tram_index, min(1, self.distance_travelled_percentage))

        if self.distance_travelled_percentage >= 1:
            self.lat = self.next_stop["stop_lat"]
            self.lon = self.next_stop["stop_lon"]
            self.current_stop_index += 1
            self.distance_travelled_percentage = 0.0
            self._print(f"made it to next stop {self.next_stop['stop_name']} ({self.current_route.stops[self.current_stop_index]})!")
            
            self.passengerManager.notify_stop_arrival(set(self.passengers), self.stopManager.get_stop(self.next_stop_id, self.direction_id))

            if self.current_route.stops[self.current_stop_index] == self.current_route.final_stop:
                self._enter_rest_state()
            else:
                self._enter_stop_state()


        else:
            self.lat = self.current_direction_vector[0]*self.distance_travelled_percentage + self.current_stop["stop_lat"]
            self.lon = self.current_direction_vector[1]*self.distance_travelled_percentage + self.current_stop["stop_lon"]

            self._print(f"travelled {time} seconds, {self.distance_travelled_percentage*100}% moving to: {self.lat},{self.lon}")


class TramManager:
    i_spawned: int = 0
    current_time: int = 0
    active_trams: list[Tram]
    print_enabled: bool = True
    routes = []


    def __init__(self, queueManager: QueueManager, stopManager: 'stop.StopManager', passengerManager: 'passenger.PassengerManager', logger: Logger, print_enabled: bool = True):
        self.print_enabled = print_enabled
        self.queueManager = queueManager
        self.stopManager = stopManager
        self.passengerManager = passengerManager
        self.logger = logger
        self.active_trams = []
        self.stopDetails = stop.StopDetails()
        self._setup()

    def _setup(self):
        self.routes = combine_routes_and_distances()

        # n_to_spawn = [1 for i in range(len(routes))]
        n_to_spawn = [4 + random.randint(1,4) for i in range(len(self.routes))]

        spawn_schedule = defaultdict(list)

        # for i, r in enumerate(routes[:1]):
        for i, r in enumerate(self.routes):
            will_spawn = n_to_spawn[i]
            if self.print_enabled: print(f"will spawn {will_spawn} of tram {i+1}")

            passed_time = 0
            direction_id = 0
            for j in range(will_spawn):
                direction_id = random.randint(0,1) if j % 2 == 0 else (1 - direction_id)
                if self.print_enabled: print(j,passed_time, direction_id)
                speed_modifier = random.uniform(0.9,1.05)
                tram_details = (r, direction_id, speed_modifier)
                spawn_schedule[passed_time].append(tram_details)

                passed_time += 60*random.randint(0,3) if j % 2 == 0 else 60*15*random.randint(1,2)

        self.sorted_spawn_times = sorted(spawn_schedule.keys())
        self.spawn_schedule = spawn_schedule

    def set_custom_spawn(self, spawn_times, spawn_schedule):
        self.sorted_spawn_times = spawn_times

        temp = defaultdict(list)
        for k in spawn_schedule.keys():
            print("yoooo", spawn_schedule[k])
            for (r_index, direction_id, speed_modifier) in spawn_schedule[k]:
                temp[k].append((self.routes[r_index], direction_id, speed_modifier))
                
        self.spawn_schedule = temp

    def spawn(self):
        if len(self.sorted_spawn_times) > 0:
            spawn_times_cleared = []
            while len(self.sorted_spawn_times) > 0 and self.sorted_spawn_times[0] <= self.current_time:
                spawn_times_cleared.append(self.sorted_spawn_times.pop(0))
            
            for s in spawn_times_cleared:
                for (r,direction_id, speed_modifier) in self.spawn_schedule[s]:
                    if self.print_enabled: print(f"spawning tram {self.i_spawned} at {self.current_time}")
                    self.active_trams.append(Tram(tram_index=self.i_spawned, direction_id=direction_id, speed_modifier=speed_modifier, stop_details=self.stopDetails, queueManager=self.queueManager, passengerManager=self.passengerManager, stopManager=self.stopManager, route=r, print_enabled=self.print_enabled, logger=self.logger))
                    self.i_spawned += 1

        res = [{"tram_index" : a.tram_index, "tram_number" : a.tram_number, "lat" : a.lat, "lon" : a.lon, "passengers": 0, "state": str(a.state)} for a in self.active_trams]

        return res

    def next_step(self) -> list[dict]:
        self.spawn()
        self.current_time += 1
        for t in self.active_trams:
            t.next_step(1)

        res = [{"tram_index" : a.tram_index, "tram_number" : a.tram_number, "lat" : a.lat, "lon" : a.lon, "passengers": len(a.passengers), "state": str(a.state), "direction": a.direction_id} for a in self.active_trams]

        return res

    def enable_print(self, print_enabled:bool):
        self.print_enabled = print_enabled
        for t in self.active_trams: t.print_enabled = print_enabled

def main():
    pass

if __name__ == "__main__":
    main ()
