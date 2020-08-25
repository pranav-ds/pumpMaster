#  Created by Pranav on 24/08/2020.
#  Copyright © 2020-2021 Pranav. All rights reserved.

# Standard Python imports
from datetime import datetime
from os import path

# Site-packages import
import numpy as np
from scipy.signal import savgol_filter
from sklearn import preprocessing


def moving_average(a, n=3):
    """Calculates the moving average of an ndarray"""
    a_new = a
    for i in range(0, n):
        a_new = np.row_stack((np.zeros(len(a[0])), a_new))
    #print(a_new)
    ret = np.cumsum(a_new, dtype=np.float32, axis=0)
    ret[n:, :] = ret[n:, :] - ret[:-n, :]
    return ret[n - 1:, :] / n


def str2date(x):
    """Convert string to date and time object"""
    return datetime.strptime(x.decode('UTF-8'), '%Y-%m-%dT%H:%M:%S')


def str2bool(x):
    """Convert string input to boolean"""
    if x.decode('UTF-8') == "NORMAL":
        return 0
    elif x.decode('UTF-8') == "BROKEN":
        return 1
    else:   # RECOVERING
        return 0


def load_sensor_data(path2file=path.abspath("../data/sensor.csv"), num_rows=100):
    """
    :param path2file: Path to the file containing sensor data
    :param num_rows : Total number of rows to read
    :return: The read data matrix
    """
    # STEP 1: Preprocess data to convert time stamps to t ~ [0.0, t_final] in seconds
    timestamps = np.loadtxt(path2file,
                            dtype='datetime64[s]',
                            delimiter=',',
                            converters={0: str2date, 4: np.float32},
                            max_rows=num_rows,
                            usecols=(1,),
                            skiprows=1)

    # An inefficient but necessary step as numpy.array subtract operaator is broken for dates
    timestamps = np.array([(timestamps[i] - timestamps[0]).item().total_seconds() for i in range(0, len(timestamps))])

    # STEP 2a: Now load sensor data
    sensor_readings = np.genfromtxt(path2file,
                                    delimiter=',',
                                    usecols=tuple(range(2, 52+2)),
                                    skip_header=1,
                                    max_rows=num_rows,
                                    missing_values='',
                                    filling_values=0.0,
                                    invalid_raise=False,
                                    usemask=False)
    # STEP 2b: Find change between sensor readings
    # sensor_readings[0, :] *= 0
    sensor_readings[1:, :] = np.diff(sensor_readings, axis=0)
    # sensor_readings[:, :] = moving_average(sensor_readings[1:, :])
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(sensor_readings[1:, :])
    sensor_readings[1:, :] = scaler.transform(sensor_readings[1:, :])

    # STEP 3: Load failure flags
    failure_signals = np.loadtxt(path2file,
                                 dtype=object,
                                 delimiter=',',
                                 converters={54: str2bool, 4: np.float32},
                                 max_rows=num_rows,
                                 usecols=(54,),
                                 skiprows=1)

    # STEP 4: Some bit of critical think over physics of the problem
    # and transforming failure_signals into a reliable continuous functions.
    # Doesn't impact regression as we are adding both LHS and RHS with \bar{x}
    failure_signals += np.mean(sensor_readings, axis=1)
    sensor_readings += np.mean(sensor_readings, axis=1).reshape(-1, 1)

    # STEP 5: Stack to create data matrix
    data_matrix = np.column_stack((timestamps, sensor_readings, failure_signals))
    data_matrix = data_matrix[1:, :]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data_matrix)
    data_matrix = scaler.transform(data_matrix)

    # STEP 6: A bit controversial to play with actual data
    # BUT filtering the raw signal is in practice useful to weed out noise and is usually done
    data_matrix[:, 1:] = savgol_filter(data_matrix[:, 1:], 3001, 2, axis=0)

    return data_matrix
