import numpy as np


class Dataset:
    def __init__(self, timestamp: np.ndarray, dataframe: np.ndarray,
                 feature_indices: list, feature_names: list,
                 target_indices: list, target_names: list):
        self.timestamp = timestamp
        self.dataframe = dataframe

        self.feature_indices = feature_indices
        self.feature_names = feature_names
        self.target_indices = target_indices
        self.target_names = target_names

        self.__check_consistency()

    def set_targets_as_features(self):
        features = self.features.copy()
        targets = self.targets.copy()

        self.dataframe = np.concatenate((features, targets, targets), axis=1)
        self.feature_indices = np.arange(0, features.shape[1] + targets.shape[1])
        self.target_indices = np.arange(features.shape[1] + targets.shape[1],
                                        features.shape[1] + targets.shape[1] +
                                        targets.shape[1])
        self.feature_names = [*self.feature_names, *self.target_names]
        for i in range(features.shape[1], features.shape[1] + targets.shape[1]):
            self.feature_names[i] = "feat - {}".format(self.feature_names[i])

        self.__check_consistency()

    def __check_consistency(self):

        # Verify that the dataframe is 2-dimensional
        assert (len(self.dataframe.shape) == 2)

        # Verify the timestamps are of the same length as the dataframe.
        assert (self.timestamp.shape[0] == self.dataframe.shape[0])

        # Verify that the indices and names have same length.
        assert (len(self.feature_indices) == len(self.feature_names))
        assert (len(self.target_indices) == len(self.target_names))

        # Verify that the indices represent valid columns
        num_cols = self.dataframe.shape[1]
        for i in self.feature_indices:
            assert (0 <= i < num_cols)
        for i in self.target_indices:
            assert (0 <= i < num_cols)

        # Verify that there are no double indices
        index_set = list({*self.feature_indices, *self.target_indices})
        assert (len(index_set) == len(self.feature_indices) + len(self.target_indices))

    @property
    def features(self) -> np.ndarray:
        return np.atleast_2d(self.dataframe[:, self.feature_indices])

    @features.setter
    def features(self, f: np.ndarray):
        assert (f.shape == self.features.shape)
        self.dataframe[:, self.feature_indices] = f

    @property
    def targets(self) -> np.ndarray:
        t = np.atleast_2d(self.dataframe[:, self.target_indices])
        if t.shape[0] == 1:
            t = t.T
        return t

    @targets.setter
    def targets(self, t: np.ndarray):
        t = np.atleast_2d(t)
        assert (t.shape == self.targets.shape)
        self.dataframe[:, self.target_indices] = t

    @property
    def num_samples(self) -> int:
        num_feature_samples = self.features.shape[0]
        num_target_samples = self.targets.shape[0]

        assert (num_feature_samples == num_target_samples)

        return num_feature_samples
