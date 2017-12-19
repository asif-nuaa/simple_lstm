import os
from datetime import datetime

import numpy as np

# Make sure to use CPU only version, as for LSTM networks it is faster than GPU based.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.optimizers import Optimizer
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.optimizers import RMSprop
from keras.models import load_model

from matplotlib import gridspec as grid
from matplotlib import pylab as plt

from sklearn.metrics import mean_squared_error

from simple_lstm import Settings
from simple_lstm import get_saver_callback


class SimpleLSTM:
    def __init__(self):

        self.start_time = datetime.now().strftime("%d.%m-%H.%M")

        # Model
        self.model = None  # type: Sequential
        self.optimizer = None  # type: Optimizer
        self.model_callbacks = []  # type: list

        self.encoding_units = [512]
        self.decoding_units = [512]
		
        num_hours_look_front = 32
        num_hours_look_back = 48
	
        num_samples_per_hour = 2
	 
        self.look_back = num_hours_look_back * num_samples_per_hour
        self.look_front = num_hours_look_front * num_samples_per_hour


        # Training
        self.lr = 0.001

        # Status
        self.status_string = "{}_look-back-{}_" \
                             "look-front-{}_units-e-{}_units-d-{}".format(
            self.start_time, self.look_back, self.look_front, self.encoding_units,
            self.decoding_units)

    def create_model(self, input_dimensionality: int, output_dimensionality: int,
                     checkpoint_path: str = None, ):
        if checkpoint_path is None:
            self.model = Sequential()
            saver_callback = get_saver_callback(checkpoint_dir=Settings.checkpoint_root,
                                                status_str=self.status_string)
            self.model_callbacks.append(saver_callback)

            # Encoder
            if len(self.encoding_units) == 1:
                self.model.add(
                    LSTM(units=self.encoding_units[0],
                         input_shape=(self.look_back, input_dimensionality),
                         return_sequences=False))
                # shape: (None, self.encoding_units[0])
            else:
                self.model.add(
                    LSTM(units=self.encoding_units[0],
                         input_shape=(self.look_back, input_dimensionality),
                         return_sequences=True))
                # shape: (None, look_back, self.encoding_units[0])

                for units in self.encoding_units[1:-1]:
                    self.model.add(
                        LSTM(units=units, return_sequences=True))
                    # shape: (None, look_back, units)

                self.model.add(
                    LSTM(units=self.encoding_units[-1], return_sequences=False))
                # shape: (None, self.encoding_units[-1])

            # Bridge between encoder and decoder.
            self.model.add(
                RepeatVector(self.look_front))
            # shape: (None, self.look_front, self.encoding_units[-1])

            # Decoder.
            for units in self.decoding_units:
                self.model.add(
                    LSTM(units=units, return_sequences=True))
                # shape: (None, self.look_front, units)

            # Readout layers (apply the same dense layer to all self.look_front matrices
            # coming from the previous layer).
            self.model.add(
                TimeDistributed(Dense(output_dimensionality)))
            # shape: (None, self.look_front, Y.shape[2])

            self.model.add(Activation("linear"))
            # shape: (None, self.look_front, Y.shape[2])

            self.optimizer = RMSprop(lr=self.lr)
            self.model.compile(loss='mse', optimizer=self.optimizer)
        else:
            # Use the checkpoint passed as argument to load the model with associated
            # weights.
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(
                    "Model checkpoint {} does not exist".format(checkpoint_path))
            self.model = load_model(checkpoint_path)
            self.optimizer = self.model.optimizer

        print(self.model.summary())

    def train(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
              test_y: np.ndarray, num_epochs: int, batch_size: int = 32):
        history = self.model.fit(train_x, train_y, epochs=num_epochs,
                                 batch_size=batch_size, validation_data=(test_x, test_y),
                                 verbose=1, shuffle=False, callbacks=self.model_callbacks)

        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def inference(self, test_x: np.ndarray) -> np.ndarray:

        print("Inference data shape: {}".format(test_x.shape))

        yhat = self.model.predict(test_x)
        print("Prediction shape: {}".format(yhat.shape))

        return yhat

    def plot_inference(self, yhat: np.ndarray, ground_truth: np.ndarray = None):
        yhat_sequential = \
            self.supervised_target_to_sequential(yhat, look_front=self.look_front)
        print("Sequential shape: {}".format(yhat_sequential.shape))
        inv_yhat = self.data_preprocessor.restore_targets(yhat_sequential)
        print("Untransformed shape: {}".format(inv_yhat.shape))
        inv_yhat = inv_yhat[:, 0]

        # Plot the average predictions.
        plt.plot(inv_yhat, label="Prediction", linewidth=3)

        # Use the ground truth data if available.
        if ground_truth is not None:
            rmse = np.sqrt(mean_squared_error(ground_truth, inv_yhat))
            print('Test RMSE: %.3f' % rmse)
            plt.plot(ground_truth, label="Ground Truth", linewidth=3)

        # Plot the single predictions.
        start = 0
        for y in yhat:
            if start % 20 == 0:
                y = self.data_preprocessor.restore_targets(y)
                plt.plot(start + np.arange(self.look_front), y)
            start += 1

        plt.legend()
        plt.show()

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
