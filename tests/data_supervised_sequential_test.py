import os

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(curr_dir, "..")
import sys

sys.path.append(root_dir)

from matplotlib import pylab as plt
from simple_lstm import Dataset

if __name__ == '__main__':

    observations = 1000
    time = np.arange(0, observations)

    features = []
    for f in range(3):
        features.append(np.linspace(f, f + 20, observations))

    features = np.array(features).T

    targets = np.concatenate((np.sin(features),), axis=1)
    targets[:, 1] *= 0.4
    targets[:, 2] *= 1.2


    dataset = Dataset(np.arange(0, len(features)),
                      np.concatenate((features, targets), axis=1),
                      feature_indices=np.arange(0, features.shape[1]),
                      target_indices=np.arange(0, targets.shape[1]) + features.shape[1],
                      feature_names=["f_{}".format(i)
                                     for i in range(features.shape[1])],
                      target_names=["t_{}".format(i) for i in range(targets.shape[1])])

    X, Y = dataset.sequential_to_supervised_data(look_back=200, look_front=300)
    x, y = Dataset.supervised_to_sequential_data(X, Y)

    f = plt.figure()
    ax = f.add_subplot(221)
    ax.plot(time, targets)
    ax.set_title("Original targ")
    ax.set_xlim([0, observations])
    ax.set_ylim([-2, 2])

    ax = f.add_subplot(222)
    ax.plot(time, targets, ":")
    ax.plot(time, y)
    ax.set_title("Restored targ")
    ax.set_xlim([0, observations])
    ax.set_ylim([-2, 2])

    ax = f.add_subplot(223)
    ax.plot(time, features)
    ax.set_title("Original feat")
    ax.set_xlim([0, observations])
    ax.set_ylim([0, 20])


    ax = f.add_subplot(224)
    ax.plot(time, features, ":")
    ax.plot(time, x)
    ax.set_title("Restored feat")
    ax.set_xlim([0, observations])
    ax.set_ylim([0, 20])


    plt.tight_layout()
    plt.show()
