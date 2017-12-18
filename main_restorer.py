import os

import numpy as np

from simple_lstm import DataPreprocessor
from simple_lstm import DataScaler
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
    return checkpoint_files[0]


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
            frequency_scale=0.05, num_samples=20000, random_seed=1, randomize=False)
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
    print("Using checkpoint {}".format(last_checkpoint_file))

    lstm = SimpleLSTM()
    dataset = load_dataset(use_csv=True, csv_file_name="oasi")  # type: Dataset
    train_fraction = 0.7
    num_train_epochs = 0

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
                                           train_fraction)  # type: DataPreprocessor

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
                      output_dimensionality=dataset.target_dimensionality,
                      checkpoint_path=last_checkpoint_file)

    if num_train_epochs > 0:
        lstm.train(train_x=X_train, train_y=Y_train, test_x=X_test, test_y=Y_test,
                   num_epochs=num_train_epochs, batch_size=32)

    predictions = lstm.inference(X_test)
    pred_x, pred_y = Dataset.supervised_to_sequential_data(X_test, predictions)

    restored_gt = data_preprocessor.restore_targets(test_y)
    restored_pred = data_preprocessor.restore_targets(pred_y)

    from matplotlib import pylab as plt

    plt.plot(restored_pred, label="Prediction")
    plt.plot(restored_gt, label="Original")
    plt.legend()
    plt.show()
