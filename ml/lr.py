#  Created by Pranav on 24/08/2020.
#  Copyright © 2020-2021 Pranav. All rights reserved.

# Standard Python imports
from os import path

# Site-packages import
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt


def build_lr(X_in,
             y_in,
             do_cv=True,
             save_cv_plots=True,
             path2save=path.abspath("../data/cv_plots"),
             *args, **kwargs):
    """
    This function builds the linear regression model
    :param X_in            : Data set
    :param y_in            : Targets
    :param do_cv           : Do cross validation
    :param save_cv_plots   : Save CV plots
    :return: The fit model
    """

    regr = linear_model.LinearRegression()

    if not do_cv:
        regr.fit(X_in, y_in)
        return regr

    else:
        # KFold splits for validation
        kf = KFold(n_splits=10)
        for index, [train_index, test_index] in enumerate(kf.split(X_in, y_in)):
            y_train = y_in[train_index].reshape(-1, 1)
            regr.fit(X_in[train_index, :], y_train)

            # Make predictions using the testing set
            y_pred = regr.predict(X_in[test_index])
            mse = mean_squared_error(y_in[test_index], y_pred)
            print("MSE: ", mse)

            if save_cv_plots:
                # Plot outputs
                fig = plt.figure()
                plt.plot(X_in[test_index, 0], y_pred, color='b', linewidth=3, label="Prediction")
                plt.plot(X_in[test_index, 0], y_in[test_index], color='r', label="Reality")

                plt.xlabel("Time")
                plt.ylabel("f(x)")
                plt.title("CV_" + str(index) + " MSE: %.3e" % (mse,))

                plt.show()
                fig.savefig(path2save + "/CV_" + str(index) + ".jpg")
                plt.close()

        return regr


