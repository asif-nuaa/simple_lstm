from datetime import datetime

import numpy as np
from matplotlib import pylab as plt

from simple_lstm import DataPreprocessor
from simple_lstm import Dataset


class PostProcessing:
    def __init__(self, dataset: Dataset, look_back: int, look_front: int,
                 predictions: np.ndarray, data_preprocessor: DataPreprocessor):
        self.dataset = dataset
        self.predictions = predictions

        # Compute the number of look backs and look fronts.
        self.look_back = look_back
        self.look_front = look_front

        self.data_preprocessor = data_preprocessor

        # Compute the number of samples predicted.
        self.num_predicted_samples = self.predictions.shape[0]

        # Extract the times corresponding to the start of each prediction.
        prediction_start_index = self.dataset.num_samples - self.num_predicted_samples \
                                 - self.look_front
        prediction_end_index = self.dataset.num_samples - self.look_front

        self.prediction_times = self.dataset.timestamp[
                                prediction_start_index: prediction_end_index].copy()

        # Extract the ground truth targets.
        self.targets = self.dataset.targets[prediction_start_index:
        prediction_end_index].copy()

        self.prediction_list = []
        self.ground_truth_list = []
        self.time_range_list = []

        self.prediction_pollution_indices = []
        self.true_pollution_indices = []

    def compute_daily_predictions(self, prediction_evaluation_hour: int):
        # Compute the indices of the predictions starting at the
        # prediction_evaluation_hour (i.e. at 16:00)
        prediction_evaluation_start_indices = []
        for i in range(1, len(self.prediction_times)):
            time_before = self.prediction_times[i - 1]  # type: datetime
            time_after = self.prediction_times[i]  # type: datetime
            if time_before.hour < prediction_evaluation_hour <= time_after.hour:
                prediction_evaluation_start_indices.append(i - 1)

        predictions_up_to_start_hour = []
        observations_to_midnight = (24 - prediction_evaluation_hour) * 2
        if observations_to_midnight >= self.look_front:
            raise RuntimeError("Cannot predict from midnight onwards at time {} only "
                               "having {} prediction steps (look front)".format(
                prediction_evaluation_hour, self.look_front))

        # Extract the predictions associated to the starting hours together with the
        # ones up to (look_front - observations_to_midnight) predictions before
        # (which will be used to average the predictions).
        # observations_to_midnight corresponds to the half-hours which are still ignored
        # between prediction_evaluation_hour (16:00) and midnight.
        for prediction_start_index in prediction_evaluation_start_indices:
            if prediction_start_index >= self.look_front - observations_to_midnight:
                predictions = self.predictions[
                              prediction_start_index + 1 - (
                                  self.look_front - observations_to_midnight):
                              prediction_start_index + 1]
                predictions_up_to_start_hour.append(predictions)
                associated_times = self.prediction_times[prediction_start_index + 1 - (
                    self.look_front - observations_to_midnight):
                prediction_start_index + 1]

            else:
                # There are no look_back predictions before the start hour.
                predictions_up_to_start_hour.append(None)

        # For each package of predictions
        # shape: ((self.look_front - observations_to_midnight) x num_features)
        # compute the mean prediction from midnight to midnight of the day after.
        mean_predictions = []
        for prediction_package in predictions_up_to_start_hour:
            if prediction_package is None:
                mean_predictions.append(None)
            else:
                mean_predictions.append(
                    self.__compute_mean_prediction(prediction_package)
                )

        # Restore the original scale of the valid predictions
        num_predictions = len(mean_predictions)
        for i in range(num_predictions):
            pred = mean_predictions[i].copy()
            if pred is None:
                continue

            start_time_index = prediction_evaluation_start_indices[i]

            ground_truth = self.targets[start_time_index + observations_to_midnight:
            start_time_index + observations_to_midnight + len(pred)]

            time_range = self.prediction_times[
                         start_time_index + observations_to_midnight:
                         start_time_index + observations_to_midnight + len(pred)]

            expected_len = self.look_front - observations_to_midnight
            if len(time_range) != expected_len \
                    or ground_truth.shape[0] != expected_len \
                    or pred.shape[0] != expected_len:
                continue

            # Restore the original scale of the predictions.
            pred = self.data_preprocessor.restore_targets(pred)

            # Restore the original scale of the ground truth.
            ground_truth = self.data_preprocessor.restore_targets(ground_truth)

            self.prediction_list.append(pred.copy())
            self.ground_truth_list.append(ground_truth.copy())
            self.time_range_list.append(time_range.copy())

        # Plot the resulting predictions
        for feature in range(self.dataset.target_dimensionality):
            f = plt.figure()
            ax = f.add_subplot(111)

            for pred, gt, time in zip(self.prediction_list, self.ground_truth_list,
                                      self.time_range_list):
                ax.plot(time, pred[:, feature], "r")
                ax.plot(time, gt[:, feature], "g")
            ax.set_title(self.dataset.target_names[feature])

        plt.show()

    @staticmethod
    def __compute_mean_prediction(prediction_package: np.ndarray) -> np.ndarray:
        # Compute the mean prediction for each target.
        num_targets = prediction_package.shape[2]
        mean_predictions = []
        for target in range(num_targets):
            predictions = prediction_package[:, :, target]
            # shape: (num_obs_before_pred_date, look_front)

            history_values_shape = (predictions.shape[0], predictions.shape[0])
            history_values = np.zeros(shape=history_values_shape, dtype=np.float32)
            history_values *= np.nan
            for i, obs in enumerate(predictions):
                history_values[i, :(i + 1)] = obs[-(i + 1):]

            # history_values has the following structure
            # [[x,0,0,0,0,...,0],
            #  [x,x,0,0,0,...,0],
            #  [x,x,x,0,0,...,0],
            #  [x,x,x,0,0,...,0],
            #  .................
            #  [x,x,x,x,x,x,x,x]]
            # Which represents all predictions from the previous num_obs_before_pred_date
            # On the time interval between midnight and midnight of the next day.

            # Compute a unique prediction by weighting the predictions in an
            # exponential way, i.e. the last row will be more important.
            lambda_weight = 0.9
            mean_prediction = np.zeros(shape=history_values.shape[1], dtype=np.float32)
            for i in range(history_values.shape[0]):
                # Extract all predictions for a single point in time.
                values = history_values[-(i + 1):, -(i + 1)]
                weighted_sum = 0
                weight_sum = 0
                for j, v in enumerate(values[::-1]):
                    weighted_sum += lambda_weight ** j * v
                    weight_sum += lambda_weight ** j
                weighted_mean = weighted_sum / weight_sum
                mean_prediction[-(i + 1)] = weighted_mean

                # If only use the last prediction (last row in history_values),
                # then uncomment this line.
                # mean_prediction[-(i+1)] = history_values[-1, -(i+1)]

            mean_predictions.append(mean_prediction)

        mean_predictions = np.array(mean_predictions).T
        return mean_predictions

    def compute_pollution_index(self):
        def __compute_NO2_index(values: np.ndarray, times: np.ndarray) -> int:
            max_NO2 = np.max(values)
            if max_NO2 <= 60:
                return 1
            elif max_NO2 <= 80:
                return 2
            elif max_NO2 <= 100:
                return 3
            elif max_NO2 <= 120:
                return 4
            elif max_NO2 <= 60:
                return 5
            return 6

        def __compute_PM10_index(values: np.ndarray, times: np.ndarray) -> int:
            max_PM10 = np.max(values)
            if max_PM10 <= 37:
                return 1
            elif max_PM10 <= 50:
                return 2
            elif max_PM10 <= 62:
                return 3
            elif max_PM10 <= 75:
                return 4
            elif max_PM10 <= 100:
                return 5
            return 6

        def __compute_O3_index(values: np.ndarray, times: np.ndarray) -> int:
            max_O3 = np.max(values)
            if max_O3 <= 90:
                return 1
            elif max_O3 <= 120:
                return 2
            elif max_O3 <= 150:
                return 3
            elif max_O3 <= 180:
                return 4
            elif max_O3 <= 240:
                return 5
            return 6

        def __compute_pollution_indices(values: np.ndarray, times: np.ndarray) -> tuple:
            index_N0 = __compute_NO2_index(values[:, 0], times)
            index_O3 = __compute_O3_index(values[:, 1], times)
            index_PM10 = __compute_PM10_index(values[:, 2], times)

            return index_N0, index_O3, index_PM10

        for pred, gt, times in zip(self.prediction_list, self.ground_truth_list,
                                   self.prediction_times):
            prediction_indices = __compute_pollution_indices(pred, times)
            true_indices = __compute_pollution_indices(gt, times)
            self.prediction_pollution_indices.append(prediction_indices)
            self.true_pollution_indices.append(true_indices)

    def create_confusion_matrix(self) -> np.ndarray:
        confusion_matrix = np.zeros(shape=(6, 6), dtype=np.int32)
        for pred_index, true_index in zip(self.prediction_pollution_indices,
                                          self.true_pollution_indices):
            p_index = np.max(pred_index)
            gt_index = np.max(true_index)

            confusion_matrix[p_index - 1, gt_index - 1] += 1

        # Flip matrix upside down, so that increasing indices go from left to right,
        # and from bottom to top.
        flipped_confusion_matrix = confusion_matrix[::-1, :]
        normalized_confusion = flipped_confusion_matrix.astype(float) / np.max(
            flipped_confusion_matrix, axis=(0, 1))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(normalized_confusion, interpolation='nearest')

        for x in range(6):
            for y in range(6):
                ax.annotate(str(flipped_confusion_matrix[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        plt.xticks(np.arange(6), np.arange(1, 7, 1))
        plt.yticks(np.arange(6), np.arange(6, 0, -1))

        ax.set_xlabel("True indices")
        ax.set_ylabel("Predicted indices")

        plt.show()

        return confusion_matrix

    def compute_errors(self):
        max_predicted_values = []
        max_true_values = []

        for pred, gt in zip(self.prediction_list, self.ground_truth_list):
            max_predicted_values.append(np.max(pred, axis=0))
            max_true_values.append(np.max(gt, axis=0))

        max_predicted_values = np.array(max_predicted_values)
        max_true_values = np.array(max_true_values)

        RMSE = np.sqrt(np.mean((max_predicted_values - max_true_values)**2, axis=0))
        MAE = np.mean(np.abs(max_predicted_values - max_true_values), axis=0)

        print("{:>5} {:>25} {:>25} {:>25}".format("", *self.dataset.target_names))
        print("{:>5} {:>25.10} {:>25.10} {:>25.10}".format("RMSE", *RMSE))
        print("{:>5} {:>25.10} {:>25.10} {:>25.10}".format("MAE", *MAE))

