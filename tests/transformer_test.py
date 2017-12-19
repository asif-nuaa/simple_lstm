import os

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(curr_dir, "..")
import sys

sys.path.append(root_dir)

from simple_lstm import DataPreprocessor, DataScaler, RelativeDifference
from matplotlib import pylab as plt

if __name__ == '__main__':
    num_samples = 40
    num_training = 20
    x_linspace = np.linspace(0, 10, num_samples)
    targets = np.sin(x_linspace)  # type: np.ndarray
    targets = targets.reshape((-1, 1))

    features = np.cos(x_linspace) + 1
    features = features.reshape((-1, 1))

    data_preprocessors = DataPreprocessor([RelativeDifference(), DataScaler()])
    data_preprocessors.fit(features[:num_training, :], targets[:num_training, :])

    plt.plot(x_linspace, features, linewidth=5, label="original_features")
    plt.plot(x_linspace, targets, linewidth=5, label="original_targets")

    features_t, targets_t = data_preprocessors.transform(features, targets)

    plt.plot(x_linspace, features_t, linewidth=3, label="tra_features")
    plt.plot(x_linspace, targets_t, linewidth=3, label="tra_targets")

    features_r, targets_r = data_preprocessors.restore(features_t, targets_t)

    plt.plot(x_linspace, features_r, linewidth=1, label="res_features")
    plt.plot(x_linspace, targets_r, linewidth=1, label="res_targets")

    plt.legend()

    plt.show()
