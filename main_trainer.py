import os

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pylab as plt

from simple_lstm import DataPreprocessor
from simple_lstm import DataScaler
from simple_lstm import Dataset
from simple_lstm import DatasetCreator, DatasetCreatorParams
from simple_lstm import DatasetLoader
from simple_lstm import Settings
from simple_lstm import SimpleLSTM


def load_dataset(use_csv: bool = True, csv_file_name: str = "oasi"):
    if use_csv:
        csv_path = os.path.join(Settings.dataset_root, csv_file_name) + ".csv"
        meta_data_path = os.path.join(Settings.dataset_root, csv_file_name) + ".json"
        dataset_loader = DatasetLoader(csv_path=csv_path,
                                       meta_data_path=meta_data_path)
        dataset = dataset_loader.load()  # type: Dataset
        return dataset
    else:
        functions = {
            lambda x: np.sin(0.3 * x),
            lambda x: 0.5 * np.cos(0.3423 * x),
            lambda x: 0.7 * np.cos(1.2 * x),
            lambda x: 1.2 * np.sin(1.45 * x)}
        dataset_creator_params = DatasetCreatorParams(
            num_features=6, num_targets=1, functions=functions, sample_dx=1.,
            frequency_scale=0.05, num_samples=10000, random_seed=1, randomize=False)
        dataset_creator = DatasetCreator(params=dataset_creator_params)
        dataset = dataset_creator.create()  # type: Dataset
        return dataset


def preprocess_dataset(dataset: Dataset, data_transformers: list,
                       preprocessing_fit_fraction: float) -> DataPreprocessor:
    data_preprocessor = DataPreprocessor([dt for dt in data_transformers])

    # Fit the data to the transformers.
    num_fit_samples = int(np.round(preprocessing_fit_fraction * dataset.num_samples))
    print("Fitting data preprocessor on {} samples of {}".format(num_fit_samples,
                                                                 dataset.num_samples))
    data_preprocessor.fit(features=dataset.features[:num_fit_samples, :],
                          targets=dataset.targets[:num_fit_samples, :])

    # Transform the entire dataset using the transformers.
    transormed_features, transformed_targets = data_preprocessor.transform(
        dataset.features, dataset.targets)

    dataset.features = transormed_features
    dataset.targets = transformed_targets

    return data_preprocessor


if __name__ == '__main__':

    lstm = SimpleLSTM()
    dataset = load_dataset(use_csv=True, csv_file_name="oasi")  # type: Dataset

    train_fraction = 0.7
    test_fraction = 1.0 - train_fraction
    preprocessing_fit_fraction = train_fraction
    num_train_epochs = 10

    use_targets_as_features = True
    if use_targets_as_features:
        dataset.set_targets_as_features()

    print("Raw data shapes:"
          "\nFeatures: {} (observations, num features)"
          "\nTargets:  {} (observations, num targets".format(
        dataset.features.shape, dataset.targets.shape))

    # Preprocessing the data
    print("Preprocessing the data")
    data_preprocessor = preprocess_dataset(dataset, [DataScaler()],
                                           preprocessing_fit_fraction)  # type: DataPreprocessor

    # Split the data into train test.
    train_x, train_y, test_x, test_y = dataset.train_test_split(
        features=dataset.features, targets=dataset.targets, train_fraction=train_fraction)

    print("Plotting the data")
    dataset.plot()

    print("Creating supervised data")
    X_train, Y_train = Dataset.sequential_to_supervised_data(
        features=train_x, targets=train_y,
        look_front=lstm.look_front, look_back=lstm.look_back)

    X_test, Y_test = Dataset.sequential_to_supervised_data(
        features=test_x, targets=test_y,
        look_front=lstm.look_front, look_back=lstm.look_back)

    print("Supervised train data shapes:"
          "\nX: {} (batch, window size, num features),"
          "\nY: {} (batch, prediction window size, num features)".format(
        X_train.shape, Y_train.shape))

    print("Supervised test data shapes:"
          "\nX: {} (batch, window size, num features),"
          "\nY: {} (batch, prediction window size, num features)".format(
        X_test.shape, Y_test.shape))

    lstm.create_model(input_dimensionality=dataset.feature_dimensionality,
                      output_dimensionality=dataset.target_dimensionality)

    if num_train_epochs > 0:
        lstm.train(train_x=X_train, train_y=Y_train, test_x=X_test, test_y=Y_test,
                   num_epochs=num_train_epochs, batch_size=32)

    predictions = lstm.inference(X_test)

    f = plt.figure()
    ax = f.add_subplot(111)

    start = 0
    for pred in predictions:
        l = pred.shape[0]
        num_feat = pred.shape[1]

        if start % 20 == 0:
            x_range = np.arange(start, l + start)

            ax.plot(x_range, pred)
        start += 1

    ax.plot(test_y, label="gt", linewidth=2.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    pred_x, pred_y = Dataset.supervised_to_sequential_data(X_test, predictions)

    restored_gt = data_preprocessor.restore_targets(test_y)
    restored_pred = data_preprocessor.restore_targets(pred_y)

    # Plot the predictions
    test_time = dataset.timestamp[-len(restored_pred):]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_tick_locator = mdates.DayLocator(interval=1)
    # Mark every 6 hours
    x_min_tick_locator = mdates.HourLocator()

    formatter = mdates.DateFormatter("%d %b '%y")

    ax.plot(test_time, restored_pred, label="Prediction")
    ax.plot(test_time, restored_gt, label="Original")
    ax.legend()

    ax.xaxis.set_major_locator(x_tick_locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.xaxis.set_minor_locator(x_min_tick_locator)

    # Plot a grid
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle=':', linewidth='1', color='black')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    # Format the coordiante box
    ax.format_xdata = mdates.DateFormatter("%d %b '%y - %H:%M")

    fig.autofmt_xdate()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
