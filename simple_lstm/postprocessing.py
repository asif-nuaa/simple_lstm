from datetime import datetime

import numpy as np

from simple_lstm import Dataset


class PostProcessing:
    def __init__(self, dataset: Dataset, test_X: np.ndarray, test_Y: np.ndarray,
                 predictions: np.ndarray):
        self.dataset = dataset
        self.test_X = test_X
        self.test_Y = test_Y
        self.predictions = predictions

        assert (self.test_Y.shape == self.predictions.shape)

        # Compute the number of look backs and look fronts.
        self.look_back = self.test_X.shape[1]
        self.look_front = self.test_Y.shape[1]

        # Compute the number of samples predicted.
        self.num_predicted_samples = self.predictions.shape[0]

        # Extract the times corresponding to the start of each prediction.
        prediction_start_index = self.dataset.num_samples - self.num_predicted_samples \
                                 - self.look_front
        prediction_end_index = self.dataset.num_samples - self.look_front
        self.prediction_times = self.dataset.timestamp[
                                prediction_start_index: prediction_end_index]

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
        # (shape: self.look_front - observations_to_midnight x num_features)
        # compute the mean prediction from midnight to midnight of the day after.
        mean_predictions = []
        for prediction_package in predictions_up_to_start_hour:
            if prediction_package is None:
                mean_predictions.append(None)
            else:
                mean_predictions.append(
                    self.__compute_mean_prediction(prediction_package)
                )

        print("Computed all mean predictions")

    def __compute_mean_prediction(self, prediction_package: np.ndarray) -> np.ndarray:
        # Compute the mean prediction for each target.
        num_targets = prediction_package.shape[2]
        mean_predictions = []
        for target in range(num_targets):
            predictions = prediction_package[:, :, target]
            # shape: (num_obs_before_pred_date, look_front)

            mean_prediction_shape = (predictions.shape[0], predictions.shape[0])
            mean_prediction = np.zeros(shape=mean_prediction_shape, dtype=np.float32)
            mean_prediction *= np.nan
            for i, obs in enumerate(predictions):
                mean_prediction[i, :(i + 1)] = obs[-(i + 1):]

            # mean prediction has the following structure
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
            single_observation = np.zeros(shape=mean_prediction.shape[1],
                                          dtype=np.float32)
            for i in range(mean_prediction.shape[0]):
                # Extract all predictions for a single point in time.
                values = mean_prediction[-(i+1):, -(i+1)]
                weighted_sum = 0
                weight_sum = 0
                for j, v in enumerate(values[::-1]):
                    weighted_sum += lambda_weight ** j * v
                    weight_sum += lambda_weight ** j
                weighted_mean = weighted_sum / weight_sum
                single_observation[-(i+1)] = weighted_mean

            mean_predictions.append(single_observation)

        mean_predictions = np.array(mean_predictions).T
        return mean_predictions
