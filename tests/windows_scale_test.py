import os

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(curr_dir, "..")
import sys

sys.path.append(root_dir)

from matplotlib import pylab as plt

from simple_lstm import Dataset


def rescale_windows(window_array: np.ndarray) -> np.ndarray:
    scaled_window_array = np.empty_like(window_array)
    for i, x in enumerate(window_array):
        new_features = []
        for j, feature in enumerate(x.T):
            if feature[0] == 0:
                feature += 1
            new_feature = [(f / feature[0]) - 1.0 for f in feature]
            new_features.append(new_feature)
        new_features = np.array(new_features)
        scaled_window_array[i, :, :] = new_features.T
    return scaled_window_array


def restore_windows(original_window_array: np.ndarray,
                    window_array: np.ndarray) -> np.ndarray:
    restored_window_array = np.empty_like(window_array)
    for i, (x_orig, x_scaled) in enumerate(zip(original_window_array, window_array)):
        new_features = []
        for j, (feature_orig, feature_scaled) in enumerate(zip(x_orig.T, x_scaled.T)):
            new_feature = [feature_orig[0] * (f + 1.0) for f in feature_scaled]
            new_features.append(new_feature)
        new_features = np.array(new_features)
        restored_window_array[i, :, :] = new_features.T
    return restored_window_array


def generate_dataframe(num_samples: int = 1000,
                       num_features: int = 2, num_targets: int = 1) -> np.ndarray:
    data = np.empty(shape=(num_samples, num_features + num_targets))
    funcs = [np.sin, np.cos, lambda x: (x / np.max(x)) ** 2 + np.sin(x)]
    for feature in range(num_features):
        x = np.arange(num_samples) / num_samples * 10.1 * np.random.normal(loc=1, scale=2)
        f = funcs[feature % len(funcs)]
        y = f(x)
        data[:, feature] = y

    for target in range(num_targets):
        y = np.zeros(shape=(num_samples))
        features = data[:, 0:num_features]
        for feature in features.T:
            y += feature * np.random.normal(size=1)
        data[:, num_features + target] = y

    return data


if __name__ == '__main__':

    num_features = 4
    num_targets = 1
    data = generate_dataframe(num_features=num_features, num_targets=num_targets)

    f = plt.figure()
    num_subplots = num_features + num_targets
    for feature in range(num_features):
        ax = f.add_subplot(num_subplots, 1, feature + 1)
        ax.plot(data[:, feature])

    for target in range(num_targets):
        ax = f.add_subplot(num_subplots, 1, num_features + target + 1)
        ax.plot(data[:, num_features + target], "r")


    x_train, y_train = Dataset.sequential_to_supervised_data(
        features=data[:, :num_features], targets=data[:, num_features:],
        look_back=20, look_front=40)

    start = 0
    shift = 0
    f = plt.figure()
    ax = f.add_subplot(111)
    for y in y_train:

        if start % 10 == 0:
            for target in y.T:
                ax.plot(np.arange(len(target)) + start, target + (shift % 8) / 10)
            shift += 1
        start += 1


    trans_x = rescale_windows(x_train)
    trans_y = rescale_windows(y_train)

    start = 0
    shift = 0
    f = plt.figure()
    ax = f.add_subplot(111)
    for y in trans_y:

        if start % 10 == 0:
            for target in y.T:
                ax.plot(np.arange(len(target)) + start, target + (shift % 8) / 10)
            shift += 1
        start += 1


    rest_x = restore_windows(x_train, trans_x)
    rest_y = restore_windows(y_train, trans_y)

    start = 0
    shift = 0
    f = plt.figure()
    ax = f.add_subplot(111)
    for y in rest_y:

        if start % 10 == 0:
            for target in y.T:
                ax.plot(np.arange(len(target)) + start, target + (shift % 8) / 10)
            shift += 1
        start += 1

    plt.show()