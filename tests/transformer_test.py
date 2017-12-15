import os

import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(curr_dir, "..")
import sys

sys.path.append(root_dir)

from simple_lstm import AbstractTransformer, DataPreprocessor, DataScaler, RelativeDifference
from matplotlib import pylab as plt


class AbsTransformer(AbstractTransformer):
    def __init__(self):
        super().__init__()

        self.__feature_signs = None  # type: np.ndarray
        self.__target_signs = None  # type: np.ndarray

    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        if self.__feature_signs.shape != features.shape \
                or self.__target_signs.shape != targets.shape:
            raise RuntimeError("You are restoring data with different shape than the "
                               "one which has been transformed!")

        targets = targets * self.__target_signs
        features = features * self.__feature_signs
        return features, targets

    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        self.__feature_signs = np.sign(features)
        self.__target_signs = np.sign(targets)

        targets = np.abs(targets)
        features = np.abs(features)
        return features, targets

    def fit(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        self._features_shape = features.shape
        self._targets_shape = features.shape

        return self.transform(features, targets)


class ShiftTransformer(AbstractTransformer):
    def __init__(self, shift: float = 0):
        super().__init__()

        self.__shift = shift

    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        features -= self.__shift
        targets -= self.__shift
        return features, targets

    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        targets += self.__shift
        features += self.__shift
        return features, targets

    def fit(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        self._features_shape = features.shape
        self._targets_shape = features.shape

        return self.transform(features, targets)


if __name__ == '__main__':
    num_samples = 40
    num_training = 20
    x_linspace = np.linspace(0, 10, num_samples)
    targets = np.sin(x_linspace)  # type: np.ndarray
    targets = targets.reshape((-1, 1))

    features = np.cos(x_linspace) + 1
    features = features.reshape((-1, 1))

    data_preprocessors = DataPreprocessor([DataScaler(), RelativeDifference(),
                                           DataScaler()])
    data_preprocessors.fit(features[:num_training, :], targets[:num_training, :])

    plt.plot(x_linspace, features, linewidth=5, label="original_features")
    # plt.plot(x_linspace, targets, linewidth=5, label="original_targets")

    features_t, targets_t = data_preprocessors.transform(features, targets)

    plt.plot(x_linspace, features_t, linewidth=3, label="tra_features")
    # plt.plot(x_linspace, targets_t, linewidth=3, label="tra_targets")

    features_r, targets_r = data_preprocessors.restore(features_t, targets_t)

    plt.plot(x_linspace, features_r, linewidth=1, label="res_features")
    # plt.plot(x_linspace, targets_r, linewidth=1, label="res_targets")

    plt.legend()

    plt.show()
