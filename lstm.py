import os

import numpy as np

# Make sure to use CPU only version, as for LSTM networks it is faster than GPU based.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Activation
from matplotlib import gridspec as grid
from matplotlib import pylab as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from simple_lstm import Settings, DatasetLoader, Dataset


class SimpleLSTM:
    def __init__(self):

        # Data
        self.use_csv_file = True
        self.dataset = None  # type: Dataset

        self.csv_path = os.path.join(Settings.dataset_root, "oasi.csv")
        self.meta_data_path = os.path.join(Settings.dataset_root, "oasi.json")

        # Preprocessing
        self.feature_transformer = None  # type: MinMaxScaler
        self.target_transformer = None  # type: MinMaxScaler
        self.smoothing_window = 1

        # Model
        self.model = None  # type: Sequential
        self.units = [128, 128]

        self.look_back = 2 * 24 * 2
        self.look_front = 1 * 24 * 2

        # Training
        self.num_epochs = 50
        self.batch_size = 32

        self.train_x = None  # type: np.ndarray
        self.train_y = None  # type: np.ndarray
        self.test_x = None  # type: np.ndarray
        self.test_y = None  # type: np.ndarray

    def run(self):
        if self.use_csv_file:
            dataset_loader = DatasetLoader(csv_path=self.csv_path,
                                           meta_data_path=self.meta_data_path)
            self.dataset = dataset_loader.load()
        else:
            self.dataset = self.create_data()

        print("Raw data shapes:"
              "\nFeatures: {} (observations, num features)"
              "\nTargets:  {} (observations, num targets".format(
            self.dataset.features.shape, self.dataset.targets.shape))

        # Preprocess the data by scaling, smoothing and shifting.
        self.preprocess_data()

        self.plot_dataframe(features=self.dataset.features, targets=self.dataset.targets,
                            feature_names=self.dataset.feature_names,
                            target_names=self.dataset.target_names)

        X, Y = self.create_supervised_data(features=self.dataset.features,
                                           targets=self.dataset.targets,
                                           look_back=self.look_back,
                                           look_front=self.look_front)

        print("Supervised data shapes:"
              "\nX: {} (batch, window size, num features),"
              "\nY: {} (batch, prediction window size, num features)".format(
            X.shape, Y.shape))

        # Split the data into train test.
        self.train_x, self.train_y, self.test_x, self.test_y = self.train_test_split(
            features=X, targets=Y, train_fraction=0.7)

        print("Train data:"
              "\n\tFeatures: {}"
              "\n\tTargets:  {}".format(self.train_x.shape, self.train_y.shape))

        print("Test data:"
              "\n\tFeatures: {}"
              "\n\tTargets:  {}".format(self.test_x.shape, self.test_y.shape))

        # Create a learning model and train in on the train data.
        self.model = Sequential()
        self.model.add(LSTM(units=self.units[0],
                            input_shape=(self.look_back, self.dataset.features.shape[1]),
                            return_sequences=False))

        self.model.add(RepeatVector(self.look_front))
        self.model.add(LSTM(units=self.units[1], return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.dataset.targets.shape[1])))
        self.model.add(Activation('linear'))

        self.model.compile(loss='mae', optimizer='adam')

        print(self.model.summary())

        history = self.model.fit(self.train_x, self.train_y, epochs=self.num_epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(self.test_x, self.test_y), verbose=1,
                                 shuffle=False)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        # invert scaling for prediction
        yhat = self.model.predict(self.test_x)
        print("Prediction shape: {}".format(yhat.shape))
        yhat_sequential = \
            self.supervised_target_to_sequential(yhat, look_front=self.look_front)
        print("Sequential shape: {}".format(yhat_sequential.shape))
        inv_yhat = self.transform_target_back(transformed_targets=yhat_sequential)
        print("Untransformed shape: {}".format(inv_yhat.shape))
        inv_yhat = inv_yhat[:, 0]

        # invert scaling for test targets
        test_y_sequential = \
            self.supervised_target_to_sequential(self.test_y, look_front=self.look_front)
        print("Y_test sequential shape: {}".format(test_y_sequential.shape))
        inv_y = self.transform_target_back(transformed_targets=test_y_sequential)
        print("Untransformed shape: {}".format(inv_y.shape))
        inv_y = inv_y[:, 0]

        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

        plt.plot(inv_yhat, label="Prediction", linewidth=2)
        plt.plot(inv_y, label="ground truth", linewidth=2)
        start = 0
        for y in yhat:
            if start % 20 == 0:
                y = self.transform_target_back(y)
                plt.plot(start + np.arange(self.look_front), y)
            start += 1

        plt.legend()
        plt.show()

    def create_data(self):
        raise RuntimeError("'create_data' function not supported yet")
        x_linspace = np.linspace(0, 150 * np.pi, 2500)

        num_features = 3
        num_targets = 1

        features = []
        functions = [np.sin, np.cos]
        for i in range(num_features):
            feature = np.random.rand() + np.random.rand() * np.random.choice(functions)(
                np.random.rand() * x_linspace + np.random.rand()
            )
            features.append(feature)
        features = np.array(features).T

        targets = []
        for i in range(num_targets):
            target = np.zeros(features.shape[0])
            for feature in features.T:
                target += np.random.rand() * feature
            targets.append(target)
        targets = np.array(targets).T

        self.dataset.dataframe = np.concatenate((features, targets), axis=1)
        self.dataset.feature_names = ["feature {}".format(i + 1) for i in range(
            self.dataset.features.shape[1])]
        self.dataset.target_names = ["target {}".format(i + 1) for i in range(
            self.dataset.targets.shape[1])]

        self.dataset.timestamp = np.arange(self.dataframe.shape[0])

    def preprocess_data(self):
        def gaussian_kernel(size: int, width: tuple = (-0.5, 0.5)) -> np.ndarray:
            k_lin = np.linspace(width[0], width[1], size)
            k = np.exp(-k_lin ** 2)
            k /= np.sum(k)
            return k

        if self.smoothing_window > 1:
            kernel = gaussian_kernel(self.smoothing_window)
            for feature_id in range(self.dataset.features.shape[1]):
                self.dataset.features[:, feature_id] = np.convolve(
                    self.dataset.features[:, feature_id],
                    kernel, "same")
            for target_id in range(self.dataset.targets.shape[1]):
                self.dataset.targets[:, target_id] = np.convolve(
                    self.dataset.targets[:, target_id],
                    kernel, "same")

        self.prepare_feature_transformer()
        self.prepare_target_transformer()

        self.dataset.features = self.transform_features(self.dataset.features)
        self.dataset.targets = self.transform_targets(self.dataset.targets)

    @staticmethod
    def create_supervised_data(features: np.ndarray, targets: np.ndarray,
                               look_back: int, look_front: int) -> tuple:
        """
        Creates a supervised representation of the data, i.e. a three dimensional array with the following dimensions:
        X.shape = (num_supervised_samples, look_back, num_features)
        Y.shape = (num_supervised_samples, look_front, num_targets)

        :param features: Numpy array (2D) of raw features (each row corresponds to one
        single time measurement)
        :param targets: Numpy array (2D) of raw targets (each row corresponds to one
        single time measurement)
        :param look_back: Number of steps to look back (memory) in the features set.
        :param look_front: Number of steps to look front (predict) in the target set.
        :return: A redundant supervised representation of the input data.
        """
        X = []  # type: list
        Y = []  # type: list

        # Need 2-dimensional data as input.
        assert (len(features.shape) == len(targets.shape) == 2)

        num_samples = features.shape[0]
        assert (num_samples == targets.shape[0])

        # Move a window of size look_back over the features predicting a successive
        # window of size look_front in the targets.
        for i in range(num_samples - look_back - look_front):
            X.append(features[i:i + look_back, :])
            Y.append(targets[i + look_back:i + look_back + look_front, :])

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
    def supervised_target_to_sequential(supervised_data: np.ndarray,
                                        look_front: int) -> np.ndarray:
        sequential_target = np.zeros(shape=(supervised_data.shape[0] + look_front - 1))
        for data_index, data in enumerate(supervised_data):
            sequential_target[data_index:data_index + look_front] += np.squeeze(data)

        # Adjust the weights in the head and tail of the predicted data (since there
        # are a lower number of predictions there due to the overlapping windows).
        for i in range(look_front):
            sequential_target[i] = sequential_target[i] * (look_front / (i + 1.0))
            sequential_target[-i - 1] = sequential_target[-i - 1] * (
                look_front / (i + 1.0))

        sequential_target = sequential_target / look_front
        return sequential_target.reshape(-1, 1)

    @staticmethod
    def plot_dataframe(features: np.ndarray, targets: np.ndarray,
                       feature_names: list, target_names: list):

        num_features = features.shape[1]
        num_targets = targets.shape[1]
        num_subplots = num_features + num_targets
        num_cols = max(num_subplots // 5, 2)
        num_rows = int(num_subplots // num_cols) + 1

        plot_height = 2.5
        plot_width = 4
        fig_size = (plot_width * num_cols, plot_height * num_rows)

        fig = plt.figure(figsize=fig_size)
        gs = grid.GridSpec(num_rows, num_cols)

        # Remove the ticks to allow more space in the plot.
        ticks_params = {'labelbottom': 'off', 'labelleft': 'off'}

        # Plot features
        for ax_index, (feature, feature_name) in enumerate(
                zip(features.T, feature_names)):
            print("plotting - {}/{} - {}({})".format(ax_index + 1, len(feature_names),
                                                     feature_name, feature.shape))

            ax = fig.add_subplot(gs[ax_index])  # type: plt.Axes
            ax.tick_params(**ticks_params)
            # Feature
            ax.plot(feature)
            ax.set_title(feature_name)

        print("Plotting targets")
        # Plot targets
        for ax_index, (target, target_name) in enumerate(zip(targets.T, target_names)):
            print("plotting - {}/{} - {}({})".format(ax_index + 1, len(feature_names),
                                                     target_name, target.shape))
            ax = fig.add_subplot(gs[ax_index + num_features])  # type: plt.Axes
            ax.tick_params(**ticks_params)
            ax.plot(target, color="red")
            ax.set_title(target_name)

        fig.suptitle("Input dataset (blue: feature, red: target)")
        gs.tight_layout(fig, rect=[0.01, 0, 0.99, 0.95])
        plt.show()

    def prepare_feature_transformer(self):
        self.feature_transformer = MinMaxScaler()
        self.feature_transformer.fit(self.dataset.features)

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        return self.feature_transformer.transform(features)

    def transform_features_back(self, transformed_features: np.ndarray) -> np.ndarray:
        return self.feature_transformer.inverse_transform(transformed_features)

    def prepare_target_transformer(self):
        self.target_transformer = MinMaxScaler()
        self.target_transformer.fit(self.dataset.targets)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_transformer.transform(targets)

    def transform_target_back(self, transformed_targets: np.ndarray) -> np.ndarray:
        return self.target_transformer.inverse_transform(transformed_targets)

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
