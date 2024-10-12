from abc import ABC, abstractmethod
import copy
import numpy as np
import typing
import math
from functools import lru_cache



Location = tuple[float, float]
PassengerGoal = tuple[int, int]

@lru_cache(maxsize=None)
def calculate_distance_haversine(location1: Location, location2: Location):
    (lon1,lat1) = location1
    (lon2,lat2) = location2


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


class Logger:
    log: list[str] = []
    time: int = 0

    def pop_log(self) -> list[str]:
        tmp = copy.copy(self.log)
        self.log.clear()
        return tmp

    def log_message(self, message: str, message_type: str = "default"):
        self.log.append({"time" : self.time, "type" : message_type, "message" : message})

    def next_step(self):
        self.time += 1



class PassengerHolder(ABC):
    holder_type: str
    holder_id: int
    holder_direction: int
    
    @abstractmethod
    def get_location(self) -> Location:
        pass

    @abstractmethod
    def add_passenger(self) -> bool:
        pass

    @abstractmethod
    def remove_passenger(self):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

def interpolate_coordinates(coordinates: list[Location], times: list[int], n: int) -> list[Location]:
    lat = [c[0] for c in coordinates]
    lon = [c[1] for c in coordinates]
    t = list(range(n))

    latint = np.interp(t, times, lat)
    lonint = np.interp(t, times, lon)

    return list(zip(latint, lonint))


if __name__ == "__main__":
    interpolate_coordinates([(45.787818,15.930054), (45.800709,15.989532), (45.809096,15.968741)], [0,50,99] ,100)