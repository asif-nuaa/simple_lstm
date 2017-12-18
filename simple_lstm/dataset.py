import numpy as np
from matplotlib import gridspec as grid
from matplotlib import pylab as plt


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
    def feature_dimensionality(self) -> int:
        return self.features.shape[1]

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
    def target_dimensionality(self) -> int:
        return self.targets.shape[1]

    @property
    def num_samples(self) -> int:
        num_feature_samples = self.features.shape[0]
        num_target_samples = self.targets.shape[0]

        assert (num_feature_samples == num_target_samples)

        return num_feature_samples

    @staticmethod
    def sequential_to_supervised_data(features: np.ndarray, targets: np.ndarray,
                                      look_back: int, look_front: int) -> tuple:
        """
        Creates a supervised representation of the data, i.e. a three dimensional array with the following dimensions:
        X.shape = (num_supervised_samples, look_back, num_features)
        Y.shape = (num_supervised_samples, look_front, num_targets)

        :param features: Numpy array of shape (num_samples, num_features) to be
        splitted in a supervised way.
        :param targets: Numpy array of shape (num_samples, num_targets) to be
        splitted in a supervised way.
        :param look_back: Number of steps to look back (memory) in the features set.
        :param look_front: Number of steps to look front (predict) in the target set.
        :return: A redundant supervised representation of the input data.
        """

        X = []  # type: list
        Y = []  # type: list

        num_samples = features.shape[0]
        assert (targets.shape[0] == num_samples)

        # Move a window of size look_back over the features predicting a successive
        # window of size look_front in the targets.
        num_supervised_samples = num_samples - look_back - look_front + 1

        for i in range(num_supervised_samples):
            feature_observation = features[i:i + look_back, :]
            target_predictions = targets[i + look_back:i + look_back + look_front, :]

            X.append(feature_observation)
            Y.append(target_predictions)

        # Vectorize the data.
        X = np.array(X)
        Y = np.array(Y)

        # Handle the case where either the features or the targets are one dimensional
        # (add a new dimension as the slices created in the sliding window are only one
        #  dimensional if there is a single dimension in the feature/target).
        if len(X.shape) == 2:
            X = X[:, :, np.newaxis]
        if len(Y.shape) == 2:
            Y = Y[:, :, np.newaxis]

        return X, Y

    @staticmethod
    def supervised_to_sequential_data(features: np.ndarray, targets: np.ndarray) -> tuple:

        look_back = features.shape[1]
        num_input_dimension = features.shape[2]
        look_front = targets.shape[1]
        num_output_dimension = targets.shape[2]

        assert (features.shape[0] == targets.shape[0])

        x = []

        for feature in features:
            x.append(feature[0, :])

        x = np.array(x)
        x = np.concatenate((x, features[-1][1:, :],
                            np.ones(shape=(look_front, num_input_dimension)) * np.nan),
                           axis=0)

        y = []
        for target in targets:
            y.append(target[0, :])

        y = np.array(y)
        y = np.concatenate((np.ones(shape=(look_back, num_output_dimension)) * np.nan,
                            y, targets[-1][1:, :]), axis=0)

        return x, y

    @staticmethod
    def train_test_split(features: np.ndarray, targets: np.ndarray,
                         train_fraction: float) -> tuple:
        assert (features.shape[0] == targets.shape[0])

        num_samples = features.shape[0]
        num_train_samples = int(np.round(train_fraction * num_samples))

        # Keep the first num_train_samples as training samples.
        train_x = features[:num_train_samples]
        train_y = targets[:num_train_samples]

        # The remaining samples for testing.
        test_x = features[num_train_samples:]
        test_y = targets[num_train_samples:]

        return train_x, train_y, test_x, test_y

    def plot(self):
        num_features = self.features.shape[1]
        num_targets = self.targets.shape[1]
        num_subplots = num_features + num_targets
        num_cols = max(num_subplots // 5, 2)
        num_rows = int(num_subplots // num_cols) + 1

        plot_height = 2.5
        plot_width = 4
        fig_size = (plot_width * num_cols, plot_height * num_rows)

        fig = plt.figure(figsize=fig_size)
        gs = grid.GridSpec(num_rows, num_cols)

        # Plot features
        for ax_index, (feature, feature_name) in enumerate(
                zip(self.features.T, self.feature_names)):
            print("plotting - {}/{} - {}({})".format(ax_index + 1,
                                                     len(self.feature_names),
                                                     feature_name, feature.shape))

            ax = fig.add_subplot(gs[ax_index])  # type: plt.Axes
            # Feature
            ax.plot(feature)
            ax.set_title(feature_name)

        print("Plotting targets")
        # Plot targets
        for ax_index, (target, target_name) in enumerate(
                zip(self.targets.T, self.target_names)):
            print("plotting - {}/{} - {}({})".format(ax_index + 1,
                                                     len(self.target_names),
                                                     target_name, target.shape))
            ax = fig.add_subplot(gs[ax_index + num_features])  # type: plt.Axes
            ax.plot(target, color="red")
            ax.set_title(target_name)

        fig.suptitle("Input Dataset (blue: feature, red: target)")
        gs.tight_layout(fig, rect=[0.01, 0, 0.99, 0.95])
        plt.show()
