#  Created by Pranav on 24/08/2020.
#  Copyright © 2020-2021 Pranav. All rights reserved.


# Standard Python imports
from os import path

# Site-packages import
import numpy as np
from matplotlib import pyplot as plt

# pumpMaster module imports
from preprocess.import_data import load_sensor_data

if __name__ == "__main__":
    # Load data
    data_matrx = load_sensor_data(path.abspath("../data/sensor_plots/sensor.csv"), num_rows=None)

    broken_here = np.sort(np.where(-1 == data_matrx[:, -1])[0])
    working_here = np.sort(np.where(1 == data_matrx[:, -1])[0])

    for j in range(1, 52 + 1):
        plt.plot(range(len(data_matrx)), data_matrx[:, j])
        plt.vlines(broken_here, 0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.savefig("Sensor_" + str(j) + ".jpg")
        plt.close()

    start = 0
    for i in range(0, len(broken_here)):
        end = broken_here[i]
        print("Starting at:", start)
        print("Ending at:", end)
        for j in range(1, 52 + 1):
            plt.plot(range(100), data_matrx[end-100:end, j], label="Sensor" + str(i))
            plt.ylim(0.0, 1.0)
            plt.savefig("Sensor_" + str(j) + "_failure_" + str(i) + ".jpg")
            plt.close()
        start = end
