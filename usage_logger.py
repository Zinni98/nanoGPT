from pynvml import nvmlInit,\
    nvmlDeviceGetHandleByIndex,\
    nvmlDeviceGetTotalEnergyConsumption, \
    nvmlDeviceGetCount

import os
import csv
import time
import datetime
import numpy as np
from enum import Enum

class Extensions(Enum):
    CSV = '.csv'
    # add here more elements if needed

class Measures(Enum):
    DEFAULT = 'default'
    EVALUATION = 'evaluation'
    TRAIN = 'train'
    FORWARD = 'forward'
    BACKWARD = 'backward'
    ATTENTION = 'attention'
    MLP = 'mlp'
    BLOCK = 'block'
    # add here more elements if needed


DEFAULT_SAVE_PATH = './logs'

class UsageLogger():
	
    def __init__(self, name=Measures.DEFAULT, save_path=DEFAULT_SAVE_PATH,
                iter_num=-1, micro_step=-1):

        self._device_nr = 0
        self._save_path = save_path
        self._name = name

        self._iter_num = iter_num
        self._micro_step = micro_step

        self._start_time = 0
        self._end_time = 0	

        self._start_energy = []
        self._end_energy = []

        self._measurements = {}

        self.__initialize()

    @property
    def name(self):
        return self._name

    # changes not allowed
    @name.setter
    def name(self):
        return

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, path):
        self._save_path = path

    @property
    def device_nr(self):
        return self._device_nr 

    @device_nr.setter
    def device_nr(self, nr):
        self._device_nr = nr

    # access not allowed
    @property
    def measurements(self):
        return self._measurements

    # changes not allowed
    @measurements.setter
    def measurements(self):
        print('External changes of the measurements are not allowed')

    def get_measurements_copy(self):
        measurements_copy = dict(self._measurements)
        return measurements_copy

    # access not allowed
    @property
    def start_energy(self):
        return

    # changes not allowed
    @start_energy.setter
    def start_energy(self):
        return

    # access not allowed
    @property
    def end_energy(self):
        return

    # changes not allowed
    @end_energy.setter
    def end_energy(self):
        return

    # access not allowed
    @property
    def start_time(self):
        return

    # changes not allowed
    @start_time.setter
    def start_time(self):
        return

    # access not allowed
    @property
    def end_time(self):
        return

    # changes not allowed
    @end_time.setter
    def end_time(self):
        return

    # access not allowed
    @property
    def iter_num(self):
        return

    # changes not allowed
    @iter_num.setter
    def iter_num(self):
        return

    # access not allowed
    @property
    def micro_step(self):
        return

    # changes not allowed
    @micro_step.setter
    def micro_step(self):
        return

	# It initializes the necessary attributes of the class based
	# on the collected information.
    def __initialize(self):

        self.__check_save_dir()

        nvmlInit()		
        self._device_nr = nvmlDeviceGetCount()
        print(f"Devices: {self._device_nr}")
        for idx in range(self._device_nr):
            key = self._name.value + '_' + str(idx)
            self._measurements[key] = []

    # this function saves the collected measurements 
    def __save_measurement(self):

        # calculate the time in milliseconds
        time = (self._end_time - self._start_time) * 1000

        # calculate the energy consumption in milliJoules
        energy = []
        for idx in range(self._device_nr):
            energy.append(round((self._end_energy[idx] - self._start_energy[idx])/1000, 4))
        
        # print(f"Saved energy consumed: {energy}")

        # append the measurements to the list
        for idx in range(self._device_nr):
            key = self._name.value + '_' + str(idx)
            self._measurements[key].append([idx, self._iter_num, self._micro_step, time, energy[idx]])

    def __check_save_dir(self):
        os.path.join(os.path.dirname(__file__), self.save_path)
        if not os.path.isdir(self.save_path):
            try:
                os.makedirs(self.save_path, exist_ok=True)

            except OSError as error:
                print(f'Error: Save directory can not be created: {self.save_path}\n')
                return None

    def __update(self, iter_num, micro_step):
        self._iter_num = iter_num
        self._micro_step = micro_step
   
    def start(self, iter_num=-1, micro_step=-1):
        self.__update(iter_num, micro_step)
        self._start_time = time.time()

        for idx in range(self._device_nr):
            handle = nvmlDeviceGetHandleByIndex(idx)
            self._start_energy.append(nvmlDeviceGetTotalEnergyConsumption(handle))

    def stop(self):
        self._end_time = time.time()

        for idx in range(self._device_nr):
            handle = nvmlDeviceGetHandleByIndex(idx)	
            self._end_energy.append(nvmlDeviceGetTotalEnergyConsumption(handle))

        self.__save_measurement()
        self.__reset()

    def __reset(self):

        self._iter_num = -1
        self._micro_step = -1

        self._start_time = 0
        self._end_time = 0	

        self._start_energy = []
        self._end_energy = []

    def export(self):

        # create header for the csv files and add it to the lists
        headers = ['device', 'iter_num', 'micro_step', 'time', 'energy']

        now = datetime.datetime.now()
        dt = now.strftime("%d%m%Y_%H%M%S") # ddmmYY_HMS
        for idx in range(self._device_nr):

            key = self._name.value + '_' + str(idx)
            self._measurements[key].insert(0, headers)

            file_name = self._name.value + '_' + str(idx) +  '_' + dt + '_' + Extensions.CSV.value
            current_save_path = os.path.join(self.save_path, self._name.value)
            if not os.path.isdir(current_save_path):
                try:
                    os.makedirs(current_save_path, exist_ok=True)
                except OSError as error:
                    print(f'Error: Save directory can not be created: {current_save_path}\n')
                    return None
            path = os.path.join(current_save_path, file_name)

            with open(path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(self._measurements[key])
    
    # def sum(self, name=Measures.DEFAULT, logger=None):
    #     if logger is None:
    #         return None

    #     new_measurements = {}

    #     logger_name = logger.name
    #     logger_measurements = logger.get_measurements_copy()

    #     for idx in range(self._device_nr):
    #         key = name.value + '_' + str(idx)
    #         logger_key = logger_name.value + '_' + str(idx)

    #         new_measurements[key] = self._measurements[key] + logger_measurements[logger_key]

    #     self._measurements = new_measurements
    #     del new_measurements


ITERATIONS_COLUMS = 3
MEASUREMENTS_COLUMNS = 5
class UsageLogger2():

    def __init__(self, logger_size, name=Measures.DEFAULT, save_path=DEFAULT_SAVE_PATH, init_NVML=True, device_nr=0):

        self._device_nr = device_nr
        self._name = name
        self._save_path = save_path

        self._iter_num = -1
        self._micro_step = -1

        self._start_time = 0
        self._end_time = 0	

        self._last_insert_idx = 0
        
        self.iterations_data = np.zeros((logger_size, ITERATIONS_COLUMS))
        
        self._start_energy = None
        self._end_energy = None

        self._measurements = None

        self.__initialize(logger_size, init_NVML, device_nr)        

    @property
    def name(self):
        return self._name

    # changes not allowed
    @name.setter
    def name(self):
        return

    @property
    def save_path(self):
        return self._save_path

    @save_path.setter
    def save_path(self, path):
        self._save_path = path

    @property
    def device_nr(self):
        return self._device_nr

    @device_nr.setter
    def device_nr(self, nr):
        self._device_nr = nr
        
    @property
    def measurements(self):
        return self._measurements

    # changes not allowed
    @measurements.setter
    def measurements(self):
        return

    # It initializes the necessary attributes of the class based
    # on the collected information.
    def __initialize(self, logger_size, init_NVML, device_nr):

        self.__check_save_dir()

        if init_NVML:
            nvmlInit()
        
        self._device_nr = nvmlDeviceGetCount() if device_nr == 0 else device_nr
            
        self._start_energy = np.zeros((logger_size, self._device_nr))
        self._end_energy = np.zeros((logger_size, self._device_nr))

        self._measurements = np.zeros((logger_size * self._device_nr, MEASUREMENTS_COLUMNS))

    # this function saves the collected measurements 
    def __save_iteration_data(self):

        time = round((self._end_time - self._start_time) * 1000, 4)
        row = np.array([self._iter_num, self._micro_step, time])
        self.iterations_data[self._last_insert_idx] = row

    def __check_save_dir(self):

        if not os.path.exists(self._save_path) and not os.path.isdir(self._save_path):
            try:
                os.makedirs(self._save_path, exist_ok=True)

            except OSError as error:
                print(f'Error: Save directory can not be created: {self._save_path}\n')
                return None

    def update(self, iter_num, micro_step=-1):

        self._iter_num = iter_num
        self._micro_step = micro_step

    def start(self):

        self._start_time = time.time()

        for idx in range(self._device_nr):
            handle = nvmlDeviceGetHandleByIndex(idx)	
            self._start_energy[self._last_insert_idx][idx] = nvmlDeviceGetTotalEnergyConsumption(handle)

    def stop(self):

        self._end_time = time.time()

        for idx in range(self._device_nr):
            handle = nvmlDeviceGetHandleByIndex(idx)	
            self._end_energy[self._last_insert_idx][idx] = nvmlDeviceGetTotalEnergyConsumption(handle)

        self.__save_iteration_data()
        self.reset()

        self._last_insert_idx += 1

    def reset(self):

        self._iter_num = -1
        self._micro_step = -1

        self._start_time = 0
        self._end_time = 0	

    def calculate(self):

        index = 0
        for iteration_data, start_energy, end_energy in zip(self.iterations_data, self._start_energy, self._end_energy):

            # i'm not sure if the transpose is necessary
            start_energy = start_energy.transpose()
            end_energy = end_energy.transpose()

            energy = (end_energy - start_energy) / 1000

            for idx in range(len(energy)):
                self.measurements[index][:len(iteration_data)] = iteration_data
                self.measurements[index][len(iteration_data):] = np.array([energy[idx], idx])
                
                index += 1

    def export(self):

        self.calculate()

        # create header for the csv files and add it to the lists
        rows = [['iter_num', 'micro_step', 'time', 'energy', 'device',]]

        now = datetime.datetime.now()
        dt = now.strftime("%d%m%Y_%H%M%S") # ddmmYY_HMS

        file_name = self._name.value +  '_' + dt + Extensions.CSV.value
        file_path = os.path.join(self._save_path, file_name)

        with open(file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for measurement in self.measurements:
                csv_writer.writerow(measurement)
