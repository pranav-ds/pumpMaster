#  Created by Pranav on 24/08/2020.
#  Copyright © 2020-2021 Pranav. All rights reserved.

# Standard Python imports
from os import path

# pumpMaster module imports
from preprocess.import_data import load_sensor_data
from ml.lr import build_lr

if __name__ == "__main__":
    # Load data
    data_matrx = load_sensor_data(path.abspath("../data/sensor.csv"), num_rows=None)

    # Extract features and response from data_matrix
    X = data_matrx[:, :-1]
    y = data_matrx[:, -1]

    # Build and validate our regression model
    regr = build_lr(X, y)