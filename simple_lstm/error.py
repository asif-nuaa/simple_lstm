import numpy as np
from sklearn.metrics import mean_squared_error as mse


def mean_squared_error(predictions: np.ndarray, ground_truth: np.ndarray) -> float:

    sample_squared_errors = []
    for pred, gt in zip(predictions, ground_truth):
        nan_location = np.logical_or(np.isnan(pred), np.isnan(gt))

        pred[nan_location] = 0
        gt[nan_location] = 0

        sample_mean_error = mse(pred, gt)
        sample_squared_errors.append(sample_mean_error * pred.size)

    sample_squared_errors = np.array(sample_squared_errors)
    return float(np.mean(sample_squared_errors))
