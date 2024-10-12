from common import Location, Logger, calculate_distance_haversine
import typing
import random
from collections import defaultdict, deque
from functools import partial
import math
import numpy as np
import time

import passenger
import stop

# includes time(int) and location
TimedLocation = tuple[int, Location]

# passenger id -> timed location
PassengerLocations = dict[int, TimedLocation]

# type stop or tram, number of stop or tram, direction of stop or tram
# type can also be None meaning unknowns -> other 2 are then also None
PassengerState = tuple[str, int, int]

Passengers = set[int]

#tram_num, direction, id
Tram = tuple[int, int, int]
TramDetails = tuple[Passengers, Location]
PredictedTrams = dict[Tram, TramDetails]

CompletePassengerState = dict[str,int,str,float,str,float,str,str,str,int,str,int]
CompletePassengerStateList = list[CompletePassengerState]
PassengerLocationLog = dict[int, CompletePassengerStateList]


class PassengerLocationLogger:
    passengerLocationLog: PassengerLocationLog = defaultdict(list)

    def add_to_log(self, passenger_id: int, location: Location, state: PassengerState, time: int):
        self.passengerLocationLog[time].append({"passenger_id": passenger_id, "lon": location[0], "lat": location[1], "holder_type": state[0], "holder_number": state[1], "holder_direction": state[2]})

    def __repr__(self) -> str:
        res = ""
        for time, passengers in self.passengerLocationLog.items():
            res += f"time:{time}\n"
            for p in passengers:
                res += f"\t{p['passenger_id']}, ({p['lat']},{p['lon']}), ({p['holder_type']}, {p['holder_number']}, {p['holder_direction']})\n"

        return res

    @property
    def json(self) -> PassengerLocationLog:
        return self.passengerLocationLog


class Predictor:
    

    time: int = 0

    def __init__(self, logger: Logger, print_enabled=True, log_enabled=True):
        self.logger = logger
        self.print_enabled = print_enabled
        self.log_enabled = log_enabled
        self.passengerLocationLogger = PassengerLocationLogger()
        self.stopRouteComparator = stop.RouteStopComparator(100)
        self.passengerLocations: PassengerLocations = defaultdict(PassengerLocations)
        self.predictedTrams: PredictedTrams = defaultdict(PredictedTrams)

        self.passengers: set['passenger.Passenger'] = set()

        # for neutral or malicious (detected) passengers, serves to sort them into proper groups
        self.passenger_location_deque = defaultdict(partial(deque, maxlen=20))
        self.passenger_first_location_map = {}
        self.passenger_update_scheduler = defaultdict(set)
        self.passenger_scheduler_map = {}

        # for grouped passengers, each group is a map from (tram_number, direction, index) -> set[passengers]
        self.passengers_in_trams = defaultdict(set)
        self.passenger_tram_map = {}

        # for groups of passenger -> predicted trams. Remembering past locations helps grouping new passengers
        self.tram_location_deque = defaultdict(partial(deque, maxlen=30))
        self.tram_indexes_map = defaultdict(set)

    @property
    def current_schedule(self):
        return math.ceil((self.time % 30) / 10)

    def check_malicious_by_history(self, points: 'stop.Points', route_to_check: tuple):

        possible_tram_routes = self.stopRouteComparator.get_possible_tram_routes(points)

    def check_malicious_by_closest_stop(self, location: Location, route_to_check: tuple):
        pass

    def _add_passenger_to_tram(self, passenger: 'passenger.Passenger', tram_key: tuple):
        self.passengers_in_trams[tram_key].add(passenger)
        self.tram_location_deque[tram_key].append((passenger.location, self.time))
        self.passenger_tram_map[passenger] = tram_key
        self.tram_indexes_map[tram_key[:-1]].add(tram_key)


    def _group_passenger(self, passenger: 'passenger.Passenger', is_new: bool):
        # new passenger will try to be grouped only if he just joined and "told" his tram/number direction
        if is_new:
            # TODO add action if detected as malicious -> most likely will be put into passenger scheduler and treated as a neutral
            self.check_malicious_by_closest_stop(passenger.location, (passenger.tram_number, passenger.tram_direction))


            any_groups = False
            current_key = (passenger.tram_number, passenger.tram_direction, 0)
            for current_key in self.tram_indexes_map[current_key[:-1]]:
                any_groups = True
                past_tram_timed_locations = self.tram_location_deque[current_key]
                last_location, last_time = past_tram_timed_locations[-1]
                if abs(self.time - last_time) >= 60:
                    if self.print_enabled: print(f"predicted tram {current_key} last location update is too old")
                else:
                    distance = calculate_distance_haversine(passenger.location, last_location)
                    if distance <= 75:
                        if self.print_enabled: print(f"passenger {passenger} probably belongs to group {current_key}")
                        self._add_passenger_to_tram(passenger, current_key)
                        return

            # basically, if passenger didn't fit in any existing groups he gets put in a new group with last index + 1
            # otherwise it means there weren't any groups to begin with so passenger get a new group with index 0
            if not any_groups: self._add_passenger_to_tram(passenger, current_key)
            else: self._add_passenger_to_tram(passenger, (*current_key[:-1], current_key[-1] + 1))
            
        # this is the case if passenger has been in the system for a while (has a location history)
        else:
            passenger_past_timed_locations = self.passenger_location_deque[passenger]
            passenger_past_locations = [i[0] for i in passenger_past_timed_locations]


            for current_key, tram_past_timed_locations in self.tram_location_deque.items():
                tram_past_locations = [i[0] for i in tram_past_timed_locations]

                route_similarity = self.stopRouteComparator.get_route_similarity(tram_past_locations, passenger_past_locations)
                if self.print_enabled: print(f"route similarty between {passenger} and {current_key} is: {route_similarity}")

                if route_similarity >= 0.9:
                    if self.print_enabled: print(f"passenger {passenger} probably belongs to group {current_key}")
                    self._add_passenger_to_tram(passenger, current_key)
                    self._passenger_schedule_remove(passenger)
                    return

            # TODO if above failed then it means passenger doesn't belong to any currently active group -> the group needs to be detected based on location history

            if len(passenger_past_locations) >= 10:
                first_location = self.passenger_first_location_map[passenger]
                (proposed_tram_route, similarity) = self.stopRouteComparator.find_closest_tram_route([first_location]+passenger_past_locations)
                if self.print_enabled: print(f"passenger {passenger} proposed tram route is {proposed_tram_route} with similarity {similarity}")
                if similarity >= 0.3:
                    # TODO NEED TO ADD INDEX SEARCH FOR PROPOSED TRAM ROUTE
                    self._add_passenger_to_tram(passenger, proposed_tram_route)
                    self._passenger_schedule_remove(passenger)

    
    def passenger_signup(self, passenger: 'passenger.Passenger'):
        if self.print_enabled: print(f"passenger {passenger.passenger_id} just signed up from {passenger.location}, time: {self.time}")
        self.passengers.add(passenger)
        self.passenger_first_location_map[passenger] = passenger.location

        if passenger.tram_number == -1:
            self.passenger_update_scheduler[self.current_schedule].add(passenger)
            self.passenger_scheduler_map[passenger] = self.current_schedule
        else:
            self._group_passenger(passenger, True)

    def passenger_signout(self, passenger: 'passenger.Passenger'):
        if self.print_enabled: print(f"passenger {passenger.passenger_id} signing out {passenger.location}, time: {self.time}")
        self.passengers.remove(passenger)
        if passenger in self.passenger_scheduler_map: self._passenger_schedule_remove(passenger)
        else: self._passenger_tram_remove(passenger)

        del self.passenger_first_location_map[passenger]


    def _passenger_tram_remove(self, passenger):
        tram_key = self.passenger_tram_map[passenger]
        self.passengers_in_trams[tram_key].remove(passenger)
        del self.passenger_tram_map[passenger]

        if len(self.passengers_in_trams[tram_key]) == 0:
            del self.passengers_in_trams[tram_key]
            del self.tram_location_deque[tram_key]
            self.tram_indexes_map[tram_key[:-1]].remove(tram_key)

    def _passenger_schedule_remove(self, passenger):
        self.passenger_update_scheduler[self.passenger_scheduler_map[passenger]].remove(passenger)
        del self.passenger_scheduler_map[passenger]


    def _passenger_update(self):
        if self.time % 10 == 0:
            passengers_to_schedule = self.passenger_update_scheduler[self.current_schedule]

            for p in set(passengers_to_schedule):
                self.passenger_location_deque[p].append((p.location, self.time))

                self._group_passenger(p, False)

    def _tram_update(self):
        if self.time % 10 == 0:
            for k in self.tram_location_deque.keys():
                passengers_in_tram = self.passengers_in_trams[k]

                random_passengers = np.random.choice(list(passengers_in_tram), max(1,len(passengers_in_tram) // 2), replace=False)
                locs = [rp.location for rp in random_passengers]

                next_tram_location = tuple(np.average(locs, axis=0))

                self.tram_location_deque[k].append((next_tram_location, self.time))


    def next_step(self) -> PredictedTrams:
        self.time += 1

        if self.log_enabled:
            for p in self.passengers:
                self.passengerLocationLogger.add_to_log(p.passenger_id, p.location, (p.holder.holder_type, p.holder.holder_id, p.holder.holder_direction), self.time)

        # t0 = time.time()

        self._passenger_update()
        # t1 = time.time()
        # pass_time = t1 - t0

        self._tram_update()
        # t2 = time.time()
        # tram_time = t2 - t1

        if self.time % 30 == 0: self._group_checker()
        # t3 = time.time()
        # group_time = t3 - t2

        # total_time = t3 - t0

        # print(f"predictor next times: {pass_time=} ({100.0 * pass_time / total_time}), {tram_time=} ({100.0 * tram_time / total_time}), {group_time=} ({100.0 * group_time / total_time}), {total_time=}")

        if self.time % 37 == 0: self._outlier_checker()

        return [{"tram_id": str(k), "lat": v[-1][0][0], "lon": v[-1][0][1], "passenger_num": len(self.passengers_in_trams[k])} for k,v in self.tram_location_deque.items()]



    def _outlier_checker(self):
        for tram, passengers in self.passengers_in_trams.items():
            passenger_locations = [p.location for p in passengers]

            possible_outliers = set()
            outlier_index =  stop.RouteStopComparator.detect_outlier_by_avg(passenger_locations, 50)
            while outlier_index is not None:
                possible_outliers.add(outlier_index)
                del passenger_locations[outlier_index]
                outlier_index =  stop.RouteStopComparator.detect_outlier_by_avg(passenger_locations, 50)
            
            if len(possible_outliers) > 0:
                print(f"possible_outliers in {tram}:", possible_outliers)
                for i, p in enumerate(list(passengers)):
                    if i in possible_outliers:
                        print(f"{p.location[0]},{p.location[1]},orange")
                        self.passengers_in_trams[tram].remove(p)
                        del self.passenger_tram_map[p]
                        self.passenger_update_scheduler[self.current_schedule].add(p)
                        self.passenger_scheduler_map[p] = self.current_schedule
                        self.logger.log_message(f"outlier detected, passenger {p} in group {tram}", "OUTLIER")
                    else:
                        print(f"{p.location[0]},{p.location[1]},pink")

            

    '''
    this a method called periodically
    it checks tram groups for similarity (in case passengers in same tram were somehow grouped to separate trams)
    if it finds 2 similar groups (with similar location history) it moves all passengers from latter to former
    and deletes all traces of former (meaning it's location history)
    '''
    def _group_checker(self):
        for num_dir,num_dir_index_list in self.tram_indexes_map.items():
            if len(num_dir_index_list) <= 1: continue

            to_skip_and_delete = set()
            seen = set()

            for num_dir_index1 in (num_dir_index_list - to_skip_and_delete - seen):
                seen.add(num_dir_index1)
                timed_locations1 = self.tram_location_deque[num_dir_index1]
                locations1 = [loc for loc,_ in timed_locations1]

                for num_dir_index2 in (num_dir_index_list - to_skip_and_delete - seen):
                    timed_locations2 = self.tram_location_deque[num_dir_index2]
                    locations2 = [loc for loc,_ in timed_locations2]

                    similarity = self.stopRouteComparator.get_route_similarity(locations1, locations2, 60)

                    if similarity > 0.85:
                        if self.print_enabled: print(f"{self.time} comparing tram groups {num_dir_index1} and {num_dir_index2} locations: {similarity}, about to get merged")
                        self._log(f"predicted tram group {num_dir_index2} was merged into {num_dir_index1} at location {locations1[-1]}")
                        passengers2 = self.passengers_in_trams[num_dir_index2]
                        self.passengers_in_trams[num_dir_index1].update(passengers2)
                        for p in passengers2: self.passenger_tram_map[p] = num_dir_index1

                        to_skip_and_delete.add(num_dir_index2)

            for num_dir_index in to_skip_and_delete:
                del self.passengers_in_trams[num_dir_index]
                del self.tram_location_deque[num_dir_index]
                self.tram_indexes_map[num_dir].remove(num_dir_index)



    def get_passenger_location_log_str(self) -> str:
        return str(self.passengerLocationLogger)


    def get_passenger_location_log_json(self) -> PassengerLocationLog:
        return self.passengerLocationLogger.json



    def _log(self, message:str, message_type: str = "PREDICTOR"):
        self.logger.log_message(f"{message}", message_type)

    def print_predicted_groups(self):
        for k, passengers in self.passengers_in_trams.items():
            last_location=self.tram_location_deque[k][-1]

            print(f"tram {k} last location: {last_location}, passengers:")
            for p in passengers:
                print(f"\t{p.location[0]},{p.location[1]}")

            print()


