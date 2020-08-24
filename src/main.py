#%%

# Importing standard libraries
import numpy as np
from datetime import datetime
from os import path

# STEP 1: Preprocess data to convert time stamps to t ~ [0.0, t_final] in seconds
str2date = lambda x: datetime.strptime(x.decode('UTF-8'), '%Y-%m-%dT%H:%M:%S')
timestamps = np.loadtxt(path.abspath("../data/sensor.csv"),
                        dtype='datetime64[s]',
                        delimiter=',',
                        converters={0: str2date, 4: np.float},
                        max_rows=100,
                        usecols=(1,),
                        skiprows=1)

#timestamps = timestamps[:, 0]

# An inefficient but necessary step as numpy.array subtract operaator is broken for dates
timestamps = np.array([(timestamps[i] - timestamps[0]).item().total_seconds() for i in range(0, len(timestamps))])

# Now load sensor data
sensor_readings = np.loadtxt(path.abspath("../data/sensor.csv"),
                             delimiter=',',
                             usecols=tuple(range(2,)),
                             skiprows=1)

print(sensor_readings)


