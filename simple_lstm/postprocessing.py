from datetime import datetime

import numpy as np
from matplotlib import pylab as plt

from simple_lstm import Dataset


class PostProcessing:
    def __init__(self, dataset: Dataset, look_back: int, look_front: int,
                 predictions: np.ndarray):
        self.dataset = dataset
        self.predictions = predictions

        # Compute the number of look backs and look fronts.
        self.look_back = look_back
        self.look_front = look_front

        # Compute the number of samples predicted.
        self.num_predicted_samples = self.predictions.shape[0]

        # Extract the times corresponding to the start of each prediction.
        prediction_start_index = self.dataset.num_samples - self.num_predicted_samples \
                                 - self.look_front
        prediction_end_index = self.dataset.num_samples - self.look_front

        print("Prediction start index: {}"
              "Prediction end index: {}".format(
            prediction_start_index, prediction_end_index))

        self.prediction_times = self.dataset.timestamp[
                                prediction_start_index: prediction_end_index]

        # Extract the ground truth targets.
        self.targets = self.dataset.targets[prediction_start_index: prediction_end_index]

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

        print("Observations between evaluation hour ({}) and midnight: {}".format(
            prediction_evaluation_hour, observations_to_midnight))

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
                print("Prediction done at {} contains the following obs".format(
                    self.prediction_times[prediction_start_index]
                ))
                print("Prediction starts: {} - Ends: {}, shape: {}".format(
                    associated_times[0], associated_times[-1],
                    predictions_up_to_start_hour[-1].shape
                ))

            else:
                # There are no look_back predictions before the start hour.
                predictions_up_to_start_hour.append(None)
                print("No prediction available for{}".format(
                    self.prediction_times[prediction_start_index]
                ))

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

        for feature in range(self.dataset.target_dimensionality):
            f = plt.figure()
            ax = f.add_subplot(111)

            num_predictions = len(mean_predictions)
            for i in range(num_predictions):
                pred = mean_predictions[i]
                if pred is None:
                    continue

                pred = pred[:, feature]

                start_time_index = prediction_evaluation_start_indices[i]
                ground_truth = self.targets[start_time_index + observations_to_midnight:
                                            start_time_index + observations_to_midnight + len(pred), feature]
                time_range = self.prediction_times[start_time_index + observations_to_midnight:
                                                   start_time_index + observations_to_midnight + len(pred)]

                expected_len = self.look_front - observations_to_midnight
                if len(time_range) != expected_len or len(ground_truth) != expected_len or len(pred) != expected_len:
                    continue

                ax.plot(time_range, pred, "r")
                ax.plot(time_range, ground_truth, "g")
            ax.set_title(self.dataset.target_names[feature])

        plt.show()

    def __compute_mean_prediction(self, prediction_package: np.ndarray) -> np.ndarray:
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
