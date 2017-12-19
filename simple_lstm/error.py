import numpy as np
from sklearn.metrics import mean_squared_error


def mean_squared_error(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    assert (len(predictions.shape) == len(ground_truth.shape) == 3)
    assert (predictions.shape == ground_truth.shape)

    sample_squared_errors = []
    for pred, gt in zip(predictions, ground_truth):
        nan_location = np.logical_or(np.isnan(pred), np.isnan(gt))
        pred[nan_location] = 0
        gt[nan_location] = 0
        sample_mean_error = mean_squared_error(pred.flatten(), gt.flatten())
        sample_squared_errors.append(sample_mean_error * pred.size)

    sample_squared_errors = np.array(sample_squared_errors)
    return np.mean(sample_squared_errors)
