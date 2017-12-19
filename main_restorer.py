import os

import matplotlib.dates as mdates
import numpy as np
from matplotlib import pylab as plt

from simple_lstm import DataPreprocessor
from simple_lstm import Dataset
from simple_lstm import DatasetCreator, DatasetCreatorParams
from simple_lstm import DatasetLoader
from simple_lstm import Settings
from simple_lstm import SimpleLSTM


def last_checkpoint(checkpoint_dir: str = Settings.checkpoint_root):
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError("Checkpoint dir {} does not exist".format(checkpoint_dir))

    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_files = [os.path.join(checkpoint_dir, c) for c in checkpoint_files]
    checkpoint_files = sorted(checkpoint_files, key=os.path.getctime, reverse=True)
    last_checkpoint_file = checkpoint_files[0]
    print("Loading checkpoint {}.".format(last_checkpoint_file))
    return last_checkpoint_file


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
                       train_fraction: float) -> DataPreprocessor:
    data_preprocessor = DataPreprocessor([dt for dt in data_transformers])

    # Fit the training data to the transformers.
    num_training_data = int(np.round(train_fraction * dataset.num_samples))
    data_preprocessor.fit(features=dataset.features[:num_training_data, :],
                          targets=dataset.targets[:num_training_data, :])

    # Transform the entire dataset using the transformers.
    transormed_features, transformed_targets = data_preprocessor.transform(
        dataset.features, dataset.targets)

    dataset.features = transormed_features
    dataset.targets = transformed_targets

    return data_preprocessor


if __name__ == '__main__':
    checkpoint_dir = Settings.checkpoint_root
    last_checkpoint_file = last_checkpoint(checkpoint_dir)
    # last_checkpoint_file = None
    print("Using checkpoint {}".format(last_checkpoint_file))

    lstm = SimpleLSTM()
    dataset = load_dataset(use_csv=True, csv_file_name="oasi")  # type: Dataset
    train_fraction = 0.9
    num_train_epochs = 0

    use_targets_as_features = True
    if use_targets_as_features:
        dataset.set_targets_as_features()

    print("Raw data shapes:"
          "\nFeatures: {} (observations, num features)"
          "\nTargets:  {} (observations, num targets".format(
        dataset.features.shape, dataset.targets.shape))

    # Preprocessing the data
    # print("Preprocessing the data")
    # data_preprocessor = preprocess_dataset(dataset, [RelativeDifference(), DataScaler()],
    #                                        train_fraction)  # type: DataPreprocessor

    print("Plotting the data")
    dataset.plot()

    # Split the data into train test.
    train_x, train_y, test_x, test_y = dataset.train_test_split(
        features=dataset.features, targets=dataset.targets, train_fraction=train_fraction)

    print("Creating supervised data")
    X_train, Y_train = Dataset.sequential_to_supervised_data(
        features=train_x, targets=train_y,
        look_front=lstm.look_front, look_back=lstm.look_back)

    X_test, Y_test = Dataset.sequential_to_supervised_data(
        features=test_x, targets=test_y,
        look_front=lstm.look_front, look_back=lstm.look_back)

    # Scale each window individually.
    def rescale_windows(window_array : np.ndarray) -> np.ndarray:
        scaled_window_array = np.empty_like(window_array)
        for i, x in enumerate(window_array):
            new_features = []
            for j, feature in enumerate(x.T):
                if feature[0] == 0:
                    feature += 1
                new_feature = [(f / feature[0]) - 1.0 for f in feature]
                new_features.append(new_feature)
            new_features = np.array(new_features)
            scaled_window_array[i, :, :] = new_features.T
        return scaled_window_array

    def restore_windows(original_window_array: np.ndarray,
                        window_array : np.ndarray) -> np.ndarray:
        restored_window_array = np.empty_like(window_array)
        for i, (x_orig, x_scaled) in enumerate(zip(original_window_array, window_array)):
            new_features = []
            for j, (feature_orig, feature_scaled) in enumerate(zip(x_orig.T, x_scaled.T)):
                new_feature = [feature_orig[0] * (f + 1.0) for f in feature_scaled]
                new_features.append(new_feature)
            new_features = np.array(new_features)
            restored_window_array[i, :, :] = new_features.T
        return restored_window_array

    X_train_scaled = rescale_windows(X_train)
    Y_train_scaled = rescale_windows(Y_train)
    X_test_scaled = rescale_windows(X_test)
    Y_test_scaled = rescale_windows(Y_test)

    print("Supervised train data shapes:"
          "\nX: {} (batch, window size, num features),"
          "\nY: {} (batch, prediction window size, num features)".format(
        X_train.shape, Y_train.shape))

    print("Supervised test data shapes:"
          "\nX: {} (batch, window size, num features),"
          "\nY: {} (batch, prediction window size, num features)".format(
        X_test.shape, Y_test.shape))

    lstm.create_model(input_dimensionality=dataset.feature_dimensionality,
                      output_dimensionality=dataset.target_dimensionality,
                      checkpoint_path=last_checkpoint_file)

    if num_train_epochs > 0:
        lstm.train(train_x=X_train, train_y=Y_train, test_x=X_test, test_y=Y_test,
                   num_epochs=num_train_epochs, batch_size=32)

    predictions = lstm.inference(X_test)


    f = plt.figure()
    ax = f.add_subplot(111)

    start = 0
    for pred, gt in predictions:
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

    # pred_x, pred_y = Dataset.supervised_to_sequential_data(X_test, predictions)
    #
    # restored_gt = data_preprocessor.restore_targets(test_y)
    # restored_pred = data_preprocessor.restore_targets(pred_y)
    #
    # # Plot the predictions
    # test_time = dataset.timestamp[-len(restored_pred):]
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # x_tick_locator = mdates.DayLocator(interval=1)
    # # Mark every 6 hours
    # x_min_tick_locator = mdates.HourLocator()
    #
    # formatter = mdates.DateFormatter("%d %b '%y")
    #
    # ax.plot(test_time, restored_pred, label="Prediction")
    # ax.plot(test_time, restored_gt, label="Original")
    # ax.legend()
    #
    # ax.xaxis.set_major_locator(x_tick_locator)
    # ax.xaxis.set_major_formatter(formatter)
    #
    # ax.xaxis.set_minor_locator(x_min_tick_locator)
    #
    # # Plot a grid
    # ax.minorticks_on()
    # # Customize the major grid
    # ax.grid(which='major', linestyle=':', linewidth='1', color='black')
    # # Customize the minor grid
    # ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    #
    # # Format the coordiante box
    # ax.format_xdata = mdates.DateFormatter("%d %b '%y - %H:%M")
    #
    # fig.autofmt_xdate()
    #
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()
