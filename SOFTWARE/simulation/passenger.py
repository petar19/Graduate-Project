from enum import Enum
import typing
import random

import tram
import stop
import predictor
from common import PassengerHolder, Location, PassengerGoal, Logger



class PassengerType(Enum):
    NEUTRAL, GOOD, MALICIOUS = range(3)

class Passenger:
    location : Location
    time: int = 0
    holder: PassengerHolder = None
    print_enabled: bool = True
    logger: Logger

    def __init__(self, gps_accuracy: float, passenger_type: PassengerType, passenger_id: int, logger: Logger, predictor: 'predictor.Predictor', print_enabled: bool = True):
        self.gps_accuracy = gps_accuracy
        self.passenger_type = passenger_type
        self.passenger_id = passenger_id
        self.print_enabled = print_enabled
        self.predictor = predictor
        self.logger = logger


    # TODO ADD MALICIOUS PASSENGER THAT RETURNS INCORRECT INFO
    @property
    def tram_number(self) -> int:
        if self.passenger_type == PassengerType.NEUTRAL or self.holder.holder_type != "tram":
            return -1
        else:
            return self.holder.holder_id

    @property
    def tram_direction(self) -> int:
        if self.passenger_type == PassengerType.NEUTRAL or self.holder.holder_type != "tram":
            return -1
        else:
            return self.holder.holder_direction

    def bind_to_holder(self, holder: PassengerHolder):
        self.holder = holder
        if self.print_enabled: print(f"passenger {self.passenger_id} just entered {self.holder}")
        self.update_location()
        self._log(f"entered {holder}", "PASSENGER")

        if holder.holder_type == "tram":
            self.predictor.passenger_signup(self)

    def exit_holder(self):
        if self.print_enabled: print(f"passenger {self.passenger_id} exiting {self.holder}")
        self._log(f"exited {self.holder}", "PASSENGER")
        self.holder.remove_passenger(self.passenger_id)

        if self.holder.holder_type == "tram":
            self.predictor.passenger_signout(self)

        self.holder = None

    def set_tram_goal(self, tram_goal: int):
        self.tram_goal = tram_goal

    def set_stop_goal(self, stop_goal: int):
        self.stop_goal = stop_goal
    
    def update_location(self):
        lat, lon = self.holder.get_location()
        self.location = (lat + self.gps_accuracy * random.uniform(-0.0005, 0.0005), lon + self.gps_accuracy * random.uniform(-0.0005, 0.0005))

    def next_step(self):
        self.update_location()
        if self.print_enabled: print(f"passenger {self.passenger_id} location: {self.location}, holder: {self.holder} {self.holder.get_location()}")
        
        self.time += 1


    def is_this_my_tram(self, tram_goal: int):
        if self.tram_goal == tram_goal:
            return True
        else: return False

    def is_this_my_stop(self, stop_goal: int):
        if self.stop_goal == stop_goal:
            return True
        else: return False

    def __repr__(self):
        return f"Passenger {self.passenger_id}, goal: ({self.tram_goal},{self.stop_goal})"

    def _log(self, message:str, message_type: str = "default"):
        self.logger.log_message(f"{self}:{message}", message_type)


class PassengerManager:
    count: int = 0
    passengers: dict[int, Passenger]
    print_enabled = True
    logger: Logger
    good_spawn_rate: float = 0.6

    def __init__(self, logger: Logger, predictor: 'predictor.Predictor', print_enabled: bool = True, good_spawn_rate: float = 0.6):
        self.print_enabled = print_enabled
        self.predictor = predictor
        self.logger = logger
        self.good_spawn_rate = good_spawn_rate
        self.passengers = {}

    @property
    def num_alive(self):
        return len(self.passengers)

    def create_new_passenger(self, goal: PassengerGoal = None) -> int:
        passenger_id = self.count
        passenger_type = PassengerType.GOOD if random.random() <= self.good_spawn_rate else PassengerType.NEUTRAL
        p = Passenger(random.random(), passenger_type, passenger_id, self.logger, self.predictor, print_enabled=self.print_enabled)
        self.count += 1

        self.passengers[passenger_id] = p

        if goal is not None:
            self.set_passenger_goal(passenger_id, goal)

        return passenger_id

    def bind_passenger(self, passenger_id: int, holder: PassengerHolder):
        assert passenger_id in self.passengers, f"Passenger {passenger_id} doesn't exist!"

        self.passengers[passenger_id].bind_to_holder(holder)

    def next_step(self) -> list[Passenger]:
        for p in self.passengers.values():
            p.next_step()

        return [{"passenger_id": p.passenger_id, "lat": p.location[0], "lon": p.location[1], "holder": str(p.holder), "goal": f"({p.tram_goal}, {p.stop_goal})", "type": str(p.passenger_type)} for p in self.passengers.values()]

    def notify_tram_arrival(self, passenger_set: set[int], tram: 'tram.Tram'):
        for p in passenger_set:
            passenger = self.passengers[p]

            if passenger.is_this_my_tram(tram.tram_number):
                if tram.add_passenger(p):
                    if self.print_enabled: print(f"passenger {p} entering tram {tram.tram_number} ({tram.tram_index})")
                    passenger.exit_holder()
                    passenger.bind_to_holder(tram)
                elif self.print_enabled: print(f"tram {tram} is apparently full since it doesn't allow passenger {p} to board on")
            elif self.print_enabled: print(f"passenger {p} not interested in {tram}")

    def notify_stop_arrival(self, passenger_set: set[int], stop: 'stop.StopObject'):
        for p in passenger_set:
            passenger = self.passengers[p]

            if passenger.is_this_my_stop(stop.stop_id):
                if self.print_enabled: print(f"passenger {p} exiting tram and getting on stop {stop.stop_id}")
                self._despawn_passenger(passenger)
                # stop.add_passenger(p) # NEED to add a chance for passenger not to despawn but choose another action like new goal

    def _despawn_passenger(self, passenger: Passenger):
        assert passenger.passenger_id in self.passengers, f"cannot despawn, {passenger} doesn't exist!"

        passenger.exit_holder()
        del self.passengers[passenger.passenger_id]

        if self.print_enabled: print(f"{passenger} despawned")

    def set_passenger_goal(self, passenger_id: int, goal: PassengerGoal):
        assert passenger_id in self.passengers, f"cannot set passenger goal, passenger {passenger_id} doesn't exist!"
        
        self.passengers[passenger_id].set_tram_goal(goal[0])
        self.passengers[passenger_id].set_stop_goal(goal[1])

    
    def set_passenger_tram_goal(self, passenger_id: int, tram_goal: int):
        assert passenger_id in self.passengers, f"cannot set passenger tram goal, passenger {passenger_id} doesn't exist!"
        
        self.passengers[passenger_id].set_tram_goal(tram_goal)

    def set_passenger_stop_goal(self, passenger_id: int, stop_goal: int):
        assert passenger_id in self.passengers, f"cannot set passenger stop goal, passenger {passenger_id} doesn't exist!"
        
        self.passengers[passenger_id].set_stop_goal(stop_goal)

    def enable_print(self, print_enabled:bool):
        self.print_enabled = print_enabled
        for p in self.passengers.values(): p.print_enabled = print_enabled




def main():
    pass

if __name__ == "__main__":
    main ()