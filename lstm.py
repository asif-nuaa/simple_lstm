import os
from datetime import datetime

import numpy as np
import pandas
from keras import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Activation
from matplotlib import pylab as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class SimpleLSTM:
    def __init__(self):

        # Data
        self.use_csv_file = False
        self.dataframe = None  # type: np.ndarray
        self.timestamp = None  # type: np.ndarray

        self.feature_names = None  # type: list
        self.target_names = None  # type: list

        # Preprocessing
        self.feature_transformer = None  # type: MinMaxScaler
        self.target_transformer = None  # type: MinMaxScaler
        self.smoothing_window = 1

        # Model
        self.model = None  # type: Sequential
        self.units = [16, 8]

        self.look_back = 200
        self.look_front = 10

        # Training
        self.num_epochs = 50
        self.batch_size = 32

        self.train_x = None  # type: np.ndarray
        self.train_y = None  # type: np.ndarray
        self.test_x = None  # type: np.ndarray
        self.test_y = None  # type: np.ndarray

    def run(self):
        if self.use_csv_file:
            self.load_csv_data()
        else:
            self.create_data()

        self.preprocess_data()

        self.plot_dataframe(timestamp=self.timestamp,
                            features=self.features, targets=self.targets,
                            feature_names=self.feature_names,
                            target_names=self.target_names)

        X, Y = self.create_supervised_data(features=self.features, targets=self.targets,
                                           look_back=self.look_back,
                                           look_front=self.look_front)

        print("Data shapes:"
              "\nX: {} (batch, window size, num features),"
              "\nY: {} (batch, prediction window size, num features)".format(
            X.shape, Y.shape))

        self.train_x, self.train_y, self.test_x, self.test_y = self.train_test_split(
            features=X, targets=Y, train_fraction=0.7)

        print("Train data:"
              "\n\tFeatures: {}"
              "\n\tTargets:  {}".format(self.train_x.shape, self.train_y.shape))

        print("Test data:"
              "\n\tFeatures: {}"
              "\n\tTargets:  {}".format(self.test_x.shape, self.test_y.shape))

        self.model = Sequential()
        self.model.add(LSTM(units=self.units[0],
                            input_shape=(self.look_back, self.features.shape[1]),
                            return_sequences=False))

        self.model.add(RepeatVector(self.look_front))
        self.model.add(LSTM(units=self.units[1], return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.targets.shape[1])))
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

    def load_csv_data(self):

        cur_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_dir = os.path.join(cur_dir, "dataset")
        dataset_file_name = os.path.join(dataset_dir, "data_clean.csv")

        self.dataframe = pandas.read_csv(dataset_file_name, delimiter=",",
                                         index_col=False)

        # Create timestamp information
        time_values = self.dataframe.values[:, 0:4].astype(np.int32).copy()
        self.dataframe.drop(labels=self.dataframe.columns[list(range(0, 4))],
                            inplace=True, axis=1)

        self.timestamp = np.array([datetime(y, m, d, h) for y, m, d, h in time_values])

        # Rename and reorder dataframe columns
        self.feature_names = self.dataframe.columns[:-1]
        self.target_names = [self.dataframe.columns[-1]]
        self.dataframe.columns = [*self.feature_names, *self.target_names]

        print(self.dataframe.head(n=10))

        # Preprocess features
        self.dataframe = self.dataframe.values.astype(np.float32)

    def create_data(self):
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

        self.dataframe = np.concatenate((features, targets), axis=1)
        self.feature_names = ["feature {}".format(i + 1) for i in range(
            self.features.shape[1])]
        self.target_names = ["target {}".format(i + 1) for i in range(
            self.targets.shape[1])]

        self.timestamp = np.arange(self.dataframe.shape[0])

    def preprocess_data(self):
        def gaussian_kernel(size: int, width: tuple = (-0.5, 0.5)) -> np.ndarray:
            k_lin = np.linspace(width[0], width[1], size)
            k = np.exp(-k_lin ** 2)
            k /= np.sum(k)
            return k

        if self.smoothing_window > 1:
            kernel = gaussian_kernel(self.smoothing_window)
            for feature_id in range(self.features.shape[1]):
                self.features[:, feature_id] = np.convolve(self.features[:, feature_id],
                                                           kernel, "same")
            for target_id in range(self.targets.shape[1]):
                self.targets[:, target_id] = np.convolve(self.targets[:, target_id],
                                                         kernel, "same")

        self.prepare_feature_transformer()
        self.prepare_target_transformer()
        self.features = self.transform_features(self.features)
        self.targets = self.transform_targets(self.targets)

    @staticmethod
    def create_supervised_data(features: np.ndarray, targets: np.ndarray,
                               look_back: int, look_front: int) -> tuple:
        X = []
        Y = []

        num_samples = features.shape[0]
        for i in range(num_samples - look_back - look_front):
            X.append(features[i:i + look_back, :])
            Y.append(targets[i + look_back:i + look_back + look_front, :])

        X = np.array(X)
        Y = np.array(Y)
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
    def plot_dataframe(timestamp: np.ndarray, features: np.ndarray, targets: np.ndarray,
                       feature_names: list, target_names: list):
        f = plt.figure(0)
        num_features = features.shape[1]
        num_targets = targets.shape[1]
        num_plots = num_features + num_targets

        # Plot features
        for ax_index, (feature, feature_name) in enumerate(zip(features.T,
                                                               feature_names)):
            ax = f.add_subplot(int(num_plots / 2 + 0.5), 2, ax_index + 1)
            # Feature
            ax.plot(timestamp, feature, label=feature_name)
            ax.legend(loc=1)

        # Plot targets
        for ax_index, (target, target_name) in enumerate(zip(targets.T, target_names)):
            ax = f.add_subplot(int(num_plots / 2 + 0.5), 2, num_features + 1 + ax_index)
            ax.plot(timestamp, target, label=target_name, color="red")
            ax.legend(loc=1)

        f.tight_layout()
        plt.show()

    def prepare_feature_transformer(self):
        self.feature_transformer = MinMaxScaler()
        self.feature_transformer.fit(self.features)

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        return self.feature_transformer.transform(features)

    def transform_features_back(self, transformed_features: np.ndarray) -> np.ndarray:
        return self.feature_transformer.inverse_transform(transformed_features)

    def prepare_target_transformer(self):
        self.target_transformer = MinMaxScaler()
        self.target_transformer.fit(self.targets)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_transformer.transform(targets)

    def transform_target_back(self, transformed_targets: np.ndarray) -> np.ndarray:
        return self.target_transformer.inverse_transform(transformed_targets)

    @property
    def features(self) -> np.ndarray:
        return np.atleast_2d(self.dataframe[:, :-1])

    @features.setter
    def features(self, f: np.ndarray):
        assert (f.shape == self.features.shape)
        self.dataframe[:, :-1] = f

    @property
    def targets(self) -> np.ndarray:
        t = np.atleast_2d(self.dataframe[:, -1])
        if t.shape[0] == 1:
            t = t.T
        return t

    @targets.setter
    def targets(self, t: np.ndarray):
        t = np.atleast_2d(t)
        assert (t.shape == self.targets.shape)
        self.dataframe[:, -1] = np.squeeze(t)

    @staticmethod
    def train_test_split(features: np.ndarray, targets: np.ndarray,
                         train_fraction: float) -> tuple:

        num_samples = features.shape[0]
        num_train_samples = int(np.round(train_fraction * num_samples))

        train_x = features[:num_train_samples]
        train_y = targets[:num_train_samples]

        test_x = features[num_train_samples:]
        test_y = targets[num_train_samples:]

        return train_x, train_y, test_x, test_y
