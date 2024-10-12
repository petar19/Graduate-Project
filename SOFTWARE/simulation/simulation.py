import random
import typing
import time
from collections import defaultdict
import pandas as pd
from fastapi import FastAPI, WebSocket
import asyncio
from abc import ABC, abstractmethod
import statistics
import json

import passenger
import tram
import stop
import predictor

from common import Logger

class Simulation():
    def __init__(self, time_delta = 15, print_enabled = True):
        self._time_delta = time_delta
        self.print_enabled = print_enabled
        self.stop_details = stop.StopDetails()
        self.i_spawned = 0
        self.current_time = 0
        self.active_trams = []

        self.setup()

    @property
    def time_delta(self):
        return self._time_delta

    @time_delta.setter
    def time_delta(self, value):
        try:
            self._time_delta = int(value)
            print(f"simulation time_delta changed to {value}")
        except ValueError:
            print(f"error while parsing new value for time_delta {value}")

    def set_seed(self, seed: int):
        if seed < 0:
            random.seed(time.time())
        else:
            random.seed(seed)

    
    def setup(self):
        random.seed(time.time())

        self.logger = Logger()
        self.predictor = predictor.Predictor(logger=self.logger, print_enabled=self.print_enabled, log_enabled=False)
        self.queueManager = tram.QueueManager()
        self.passengerManager = passenger.PassengerManager(logger=self.logger, predictor=self.predictor, print_enabled=self.print_enabled)
        self.stopManager = stop.StopManager(self.stop_details, self.passengerManager, logger=self.logger, print_enabled=self.print_enabled)
        self.tramManager = tram.TramManager(queueManager = self.queueManager, stopManager = self.stopManager, passengerManager=self.passengerManager, logger=self.logger, print_enabled=self.print_enabled)

    def next_step_one_second(self):

        predictedTrams = []
        passengers = []
        trams = []
        for _ in range(self._time_delta):
            self.logger.next_step()
            self.current_time += 1
            trams = self.tramManager.next_step()
            self.stopManager.next_step()
            passengers = self.passengerManager.next_step()
            predictedTrams = self.predictor.next_step()


        log = self.logger.pop_log()
        res = (self.current_time,
                trams, passengers, predictedTrams, log)

        return res

    def start(self):
        log = self.logger.pop_log()

        trams = self.tramManager.spawn()
        
        res = (self.current_time, trams, log)

        return res

    def __repr__(self):
        res = "checking state"
        res += f"{len(self.tramManager.active_trams)=}\n"
        res += f"{len(self.passengerManager.passengers)=}\n"
        res += f"{len(self.predictor.passengers_in_trams)=}\n"

        return res


    def benchmark(self, duration: int):
        tram_times = []
        stop_times = []
        passenger_times = []
        prediction_times = []
        total_times = []

        def print_stats():
            total_time = sum(total_times)

            print(f"tram mean time: {statistics.mean(tram_times)}, tram median time: {statistics.median(tram_times)}, percentage: {100 * sum(tram_times) / total_time}%")
            print(f"stop mean time: {statistics.mean(stop_times)}, stop median time: {statistics.median(stop_times)}, percentage: {100 * sum(stop_times) / total_time}%")
            print(f"passenger mean time: {statistics.mean(passenger_times)}, passenger median time: {statistics.median(passenger_times)}, percentage: {100 * sum(passenger_times) / total_time}%")
            print(f"prediction mean time: {statistics.mean(prediction_times)}, prediction median time: {statistics.median(prediction_times)}, percentage: {100 * sum(prediction_times) / total_time}%")
            print(f"total mean time: {statistics.mean(total_times)}, total median time: {statistics.median(total_times)}, {total_time}")

        t0 = time.time()
        active_passengers = 0

        for _ in range(duration):
            self.current_time += 1
            if self.current_time % 500 == 0:
                print(f"benchmark {100 * self.current_time / duration}%, {time.time() - t0}s, active passengers: {active_passengers}")
                print_stats()
                print()

            t1 = time.time()

            self.tramManager.next_step()
            t2 = time.time()
            tram_times.append(t2-t1)

            self.stopManager.next_step()
            t3 = time.time()
            stop_times.append(t3-t2)
            active_passengers = len(self.passengerManager.next_step())
            t4 = time.time()
            passenger_times.append(t4-t3)
            predictions = self.predictor.next_step()
            t5 = time.time()
            prediction_times.append(t5-t4)
            total_times.append(t5-t1)

        print(f"BENCHMARK DONE, time:{time.time() - t0}s, active passengers: {active_passengers}")
        print_stats()

        total_time = sum(total_times)
        factor = duration / total_time
        tram_time = 100 * sum(tram_times) / total_time
        stop_time = 100 * sum(stop_times) / total_time
        passenger_time = 100 * sum(passenger_times) / total_time
        prediction_time= 100 * sum(prediction_times) / total_time
        print(f"{duration} & {self.stopManager.max_passenger_num} & {self.passengerManager.good_spawn_rate*100:.0f} & {total_time:.1f} & {factor:.1f} & {tram_time:.1f} & {stop_time:.1f} & {passenger_time:.1f} & {prediction_time:.1f}")

    def print_passenger_location_log(self):
        print(self.predictor.get_passenger_location_log_str())

    def save_passenger_location_log(self, file_name: str = "passenger_location_log.json"):
        with open(file_name, "w") as file:
            json.dump(self.predictor.get_passenger_location_log_json(),file,indent=4)

    def set_time_delta(self, time_delta: int):
        if time_delta >= 1 and time_delta <= 300:
            self._time_delta = time_delta
            print(f"changed {self._time_delta=}")

    def set_passenger_num(self, passenger_num: int):
        if passenger_num >= 0 and passenger_num <= 100000:
            self.stopManager.max_passenger_num = passenger_num
            print(f"changed {self.stopManager.max_passenger_num=}")

    def set_good_spawn_rate(self, good_spawn_rate: float):
        if good_spawn_rate >= 0 and good_spawn_rate <= 1:
            self.passengerManager.good_spawn_rate = good_spawn_rate
            print(f"changed {self.passengerManager.good_spawn_rate=}")

    

    def print_predicted_groups(self):
        self.predictor.print_predicted_groups()

    def enable_print(self, print_enabled:bool):
        self.print_enabled = print_enabled
        print(f"changed {self.print_enabled=}")
        self.predictor.print_enabled = print_enabled
        self.queueManager.print_enabled = print_enabled

        self.stopManager.enable_print(print_enabled)
        self.tramManager.enable_print(print_enabled)
        self.passengerManager.enable_print(print_enabled)

    def start_test_case(self, case: str = "default"):
        log = self.logger.pop_log()

        if case == "default":
            pass
        elif case == "test1":
            self.set_passenger_num(100)
            self.stopManager.set_custom_spawn_viable_stops([(317,0), (103,0), (169,0), (105,0), (106,0), (296,0), (299,0), (163,0), (304,0), (220,0), (292,0), (217,0), (291,0), (193,0), (193,1), (291,1), (217,1), (292,1), (220,1), (304,1), (163,1), (299,1), (296,1), (106,1), (105,1), (169,1), (103,1), (293,1), (317,1)])
            
            tram_spawn_times = [0, 30]
            tram_spawn_schedule = {0: [(0, 0, 1)], 30: [(0, 1, 1.03)]}
            self.tramManager.set_custom_spawn(tram_spawn_times, tram_spawn_schedule)

        elif case == "test_overtaking":
            self.set_passenger_num(0)
            
            tram_spawn_times = [0, 30]
            tram_spawn_schedule = {0: [(0, 0, 1)], 30: [(0, 0, 1.5)]}
            self.tramManager.set_custom_spawn(tram_spawn_times, tram_spawn_schedule)

        elif case == "test2":
            self.set_passenger_num(20)
            self.stopManager.set_custom_spawn_viable_stops([(111,0), (111,1), (112, 0), (112, 1)])

            tram_spawn_times = [0, 30]
            tram_spawn_schedule = {0: [(0, 0, 1)], 30: [(0, 0, 1.5)]}
            self.tramManager.set_custom_spawn(tram_spawn_times, tram_spawn_schedule)

        trams = self.tramManager.spawn()
        res = (self.current_time, trams, log)

        return res



app = FastAPI()
sd = stop.StopDetails()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print_enabled = False
    if print_enabled: print('Accepting client connection...')
    await websocket.accept()
    simulation = None
    while True:
        try:
            # Wait for any message from the client
            data = await websocket.receive_text()
            if print_enabled: print(f"got message from client: {data}")

            if data.startswith("start_simulation"):
                if print_enabled: print("client is starting the simulation")
                resp = {"message_type" : "stops", "message_data" : sd.get_all_stops()}
                print("before starting", simulation)
                simulation = Simulation(time_delta=15, print_enabled=print_enabled)
                print("after starting", simulation)
                await websocket.send_json(resp)

                current_time, tram_data, log = simulation.start()
                resp = {"message_type" : "update", "message_data" : {"current_time" : current_time, "trams": tram_data, "passengers": [], "log": log, "predicted_trams" : []}}
                await websocket.send_json(resp)

            elif data.startswith("test"):
                if "," not in data or data.split(",")[1] == '':
                    resp = {"message_type" : "error", "message_data" : "missing value for time_delta"}
                else:
                    test_type = data.split(",")[1]
                    print(f"about to run {test_type=}")

                    resp = {"message_type" : "stops", "message_data" : sd.get_all_stops()}
                    simulation = Simulation(time_delta=15, print_enabled=print_enabled)
                    await websocket.send_json(resp)

                    current_time, tram_data, log = simulation.start_test_case(test_type)
                    resp = {"message_type" : "update", "message_data" : {"current_time" : current_time, "trams": tram_data, "passengers": [], "log": log, "predicted_trams" : []}}
                    await websocket.send_json(resp)

            elif data.startswith("seed"):
                if "," not in data or data.split(",")[1] == '':
                    resp = {"message_type" : "error", "message_data" : "missing value for time_delta"}
                else:
                    seed = int(data.split(",")[1])
                    simulation.set_seed(seed)

            elif data.startswith("next"):
                if print_enabled: print("client wants next step in simulation")

                if simulation is None:
                    if print_enabled: print("error, simulation hasn't been started yet")
                    resp = {"message_type" : "error", "message_data" : "error, simulation hasn't been started yet"}
                    await websocket.send_json(resp)
                else:
                    current_time, tram_data, passenger_data, predicted_trams, log = simulation.next_step_one_second()
                    resp = {"message_type" : "update", "message_data" : {"current_time" : current_time, "trams": tram_data, "passengers": passenger_data, "log": log, "predicted_trams" : predicted_trams}}
                    await websocket.send_json(resp)
            
            elif data.startswith("time_delta"):
                if "," not in data or data.split(",")[1] == '':
                    resp = {"message_type" : "error", "message_data" : "missing value for time_delta"}
                else:
                    time_delta = int(data.split(",")[1])
                    simulation.set_time_delta(time_delta)

            elif data.startswith("passenger_num"):
                if "," not in data or data.split(",")[1] == '':
                    resp = {"message_type" : "error", "message_data" : "missing value for passenger_num"}
                else:
                    passenger_num = int(data.split(",")[1])
                    simulation.set_passenger_num(passenger_num)

            elif data.startswith("print_enabled"):
                if "," not in data or data.split(",")[1] == '':
                    resp = {"message_type" : "error", "message_data" : "missing value for print_enabled"}
                else:
                    pe = bool(int(data.split(",")[1]))
                    simulation.enable_print(pe)

            elif data.startswith("good_spawn_rate"):
                if "," not in data or data.split(",")[1] == '':
                    resp = {"message_type" : "error", "message_data" : "missing value for print_enabled"}
                else:
                    good_spawn_rate = float(data.split(",")[1])
                    simulation.set_good_spawn_rate(good_spawn_rate)

            elif data.startswith("print_groups"):
                simulation.print_predicted_groups()



        except Exception as e:
            print('error:', e)
            raise e
            break
    print('Bye..')

def sim_test():
    s = Simulation(print_enabled=False)

    s.set_passenger_num(500)
    s.set_good_spawn_rate(0.5)
    s.benchmark(10000)

    # for _ in range(100): s.next_step_one_second()
    # s.enable_print(True)
    # for _ in range(100): s.next_step_one_second()
    # s.enable_print(False)
    # for _ in range(100): s.next_step_one_second()
    # s.enable_print(True)


    # s.save_passenger_location_log()

def main():
    sim_test()



if __name__ == "__main__":
    main()
