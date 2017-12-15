import os
from datetime import datetime

import numpy as np

# Make sure to use CPU only version, as for LSTM networks it is faster than GPU based.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.optimizers import RMSprop

from matplotlib import gridspec as grid
from matplotlib import pylab as plt

from sklearn.metrics import mean_squared_error

from simple_lstm import Settings
from simple_lstm import DatasetLoader
from simple_lstm import Dataset
from simple_lstm import DatasetCreatorParams
from simple_lstm import DatasetCreator
from simple_lstm import get_saver_callback
from simple_lstm import DataPreprocessor
from simple_lstm import DataScaler
from simple_lstm import RelativeDifference


class SimpleLSTM:
    def __init__(self):

        self.start_time = datetime.now().strftime("%d.%m-%H.%M")

        # Data
        self.use_csv_file = False
        self.dataset = None  # type: Dataset
        self.use_targets_as_feature = True

        self.csv_path = os.path.join(Settings.dataset_root, "oasi.csv")
        self.meta_data_path = os.path.join(Settings.dataset_root, "oasi.json")

        # Preprocessing
        self.data_preprocessor = None  # type: DataPreprocessor
        self.transfomers = [RelativeDifference(), DataScaler()]

        # Model
        self.model = None  # type: Sequential
        self.model_callbacks = []  # type: list

        self.encoding_units = [256]
        self.decoding_units = [256]

        self.look_back = int(2 * 24 * 2)
        self.look_front = int(1 * 24 * 2)

        # Training
        self.num_epochs = 1000
        self.batch_size = 32
        self.train_fraction = 0.7
        self.lr = 0.0002

        self.train_x = None  # type: np.ndarray
        self.train_y = None  # type: np.ndarray
        self.test_x = None  # type: np.ndarray
        self.test_y = None  # type: np.ndarray

        # Status
        self.status_string = "{}_use-targets-{}_look-back-{}_" \
                             "look-front-{}_units-e-{}_units-d-{}".format(
            self.start_time, self.use_targets_as_feature,
            self.look_back, self.look_front, self.encoding_units, self.decoding_units)

    def run(self):
        if self.use_csv_file:
            dataset_loader = DatasetLoader(csv_path=self.csv_path,
                                           meta_data_path=self.meta_data_path)
            self.dataset = dataset_loader.load()
        else:
            functions = {
                lambda x: np.sin(0.3 * x),
                lambda x: 0.5 * np.cos(0.3423 * x),
                lambda x: 0.7 * np.cos(1.2 * x),
                lambda x: 1.2 * np.sin(1.45 * x)}
            dataset_creator_params = DatasetCreatorParams(
                num_features=3, num_targets=1, functions=functions, sample_dx=1.,
                frequency_scale=0.05, num_samples=10000, random_seed=1, randomize=False)
            dataset_creator = DatasetCreator(params=dataset_creator_params)
            self.dataset = dataset_creator.create()

        if self.use_targets_as_feature:
            self.dataset.set_targets_as_features()

        print("Raw data shapes:"
              "\nFeatures: {} (observations, num features)"
              "\nTargets:  {} (observations, num targets".format(
            self.dataset.features.shape, self.dataset.targets.shape))

        self.plot_dataframe(features=self.dataset.features, targets=self.dataset.targets,
                            feature_names=self.dataset.feature_names,
                            target_names=self.dataset.target_names)

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
            features=X, targets=Y, train_fraction=self.train_fraction)

        print("Train data:"
              "\n\tFeatures: {}"
              "\n\tTargets:  {}".format(self.train_x.shape, self.train_y.shape))

        print("Test data:"
              "\n\tFeatures: {}"
              "\n\tTargets:  {}".format(self.test_x.shape, self.test_y.shape))

        # Create a learning model and train in on the train data.

        print("Encoding units: {}".format(self.encoding_units))
        print("Decoding units: {}".format(self.decoding_units))
        print("Look back: {}".format(self.look_back))
        print("Features in: {}".format(X.shape[2]))
        print("Look front: {}".format(self.look_front))
        print("Features out: {}".format(Y.shape[2]))

        self.model = Sequential()
        saver_callback = get_saver_callback(checkpoint_dir=Settings.checkpoint_root,
                                            status_str=self.status_string)
        self.model_callbacks.append(saver_callback)

        # Encoder
        if len(self.encoding_units) == 1:
            self.model.add(LSTM(units=self.encoding_units[0],
                                input_shape=(
                                    self.look_back, X.shape[2]),
                                return_sequences=False))
            # shape: (None, self.encoding_units[0])
        else:
            self.model.add(LSTM(units=self.encoding_units[0],
                                input_shape=(
                                    self.look_back, X.shape[2]),
                                return_sequences=True))
            # shape: (None, look_back, self.encoding_units[0])

            for units in self.encoding_units[1:-1]:
                self.model.add(LSTM(units=units, return_sequences=True))
                # shape: (None, look_back, units)

            self.model.add(LSTM(units=self.encoding_units[-1], return_sequences=False))
            # shape: (None, self.encoding_units[-1])

        # Bridge between encoder and decoder.
        self.model.add(RepeatVector(self.look_front))
        # shape: (None, self.look_front, self.encoding_units[-1])

        # Decoder.
        for units in self.decoding_units:
            self.model.add(LSTM(units=units, return_sequences=True))
            # shape: (None, self.look_front, units)

        # Readout layers (apply the same dense layer to all self.look_front matrices
        # coming from the previous layer).
        self.model.add(TimeDistributed(Dense(Y.shape[2])))
        # shape: (None, self.look_front, Y.shape[2])

        self.model.add(Activation("linear"))
        # shape: (None, self.look_front, Y.shape[2])

        self.optimizer = RMSprop(lr=self.lr)
        self.model.compile(loss='mse', optimizer=self.optimizer)

        print(self.model.summary())

        history = self.model.fit(self.train_x, self.train_y, epochs=self.num_epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(self.test_x, self.test_y), verbose=1,
                                 shuffle=False, callbacks=[saver_callback])

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        # invert scaling for prediction
        yhat = self.model.predict(self.test_x)
        print("Mean targets: {}".format(np.mean(yhat, axis=(1, 2))))
        print("Prediction shape: {}".format(yhat.shape))
        yhat_sequential = \
            self.supervised_target_to_sequential(yhat, look_front=self.look_front)
        print("Sequential shape: {}".format(yhat_sequential.shape))
        inv_yhat = self.data_preprocessor.restore_targets(yhat_sequential)
        print("Untransformed shape: {}".format(inv_yhat.shape))
        inv_yhat = inv_yhat[:, 0]

        # invert scaling for test targets
        test_y_sequential = \
            self.supervised_target_to_sequential(self.test_y, look_front=self.look_front)
        print("Y_test sequential shape: {}".format(test_y_sequential.shape))
        inv_y = self.data_preprocessor.restore_targets(test_y_sequential)
        print("Untransformed shape: {}".format(inv_y.shape))
        inv_y = inv_y[:, 0]

        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

        plt.plot(inv_yhat, label="Prediction", linewidth=3)
        plt.plot(inv_y, label="ground truth", linewidth=3)
        start = 0
        for y in yhat:
            if start % 20 == 0:
                y = self.data_preprocessor.restore_targets(y)
                plt.plot(start + np.arange(self.look_front), y)
            start += 1

        plt.legend()
        plt.show()

    def preprocess_data(self):

        self.data_preprocessor = DataPreprocessor(self.transfomers)

        # Fit the training data to the transformers.
        num_training_data = int(np.round(self.train_fraction * self.dataset.num_samples))
        self.data_preprocessor.fit(features=self.dataset.features[:num_training_data, :],
                                   targets=self.dataset.targets[:num_training_data, :])

        # Transform the entire dataset using the transformers.
        transormed_features, transformed_targets = self.data_preprocessor.transform(
            self.dataset.features, self.dataset.targets)

        self.dataset.features = transormed_features
        self.dataset.targets = transformed_targets

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

        # Plot features
        for ax_index, (feature, feature_name) in enumerate(
                zip(features.T, feature_names)):
            print("plotting - {}/{} - {}({})".format(ax_index + 1, len(feature_names),
                                                     feature_name, feature.shape))

            ax = fig.add_subplot(gs[ax_index])  # type: plt.Axes
            # Feature
            ax.plot(feature)
            ax.set_title(feature_name)

        print("Plotting targets")
        # Plot targets
        for ax_index, (target, target_name) in enumerate(zip(targets.T, target_names)):
            print("plotting - {}/{} - {}({})".format(ax_index + 1, len(feature_names),
                                                     target_name, target.shape))
            ax = fig.add_subplot(gs[ax_index + num_features])  # type: plt.Axes
            ax.plot(target, color="red")
            ax.set_title(target_name)

        fig.suptitle("Input dataset (blue: feature, red: target)")
        gs.tight_layout(fig, rect=[0.01, 0, 0.99, 0.95])
        plt.show()

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
