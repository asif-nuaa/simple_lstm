import os

from simple_lstm import Settings
from simple_lstm import SimpleLSTM


def last_checkpoint(checkpoint_dir: str = Settings.checkpoint_root):
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError("Checkpoint dir {} does not exist".format(checkpoint_dir))

    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_files = [os.path.join(checkpoint_dir, c) for c in checkpoint_files]
    checkpoint_files = sorted(checkpoint_files, key=os.path.getctime, reverse=True)
    return checkpoint_files[0]


if __name__ == '__main__':
    checkpoint_dir = os.path.join(os.path.join(Settings.simple_lstm_root, os.pardir),
                                  "checkpoint_save")
    last_checkpoint_file = last_checkpoint(checkpoint_dir)
    print("Using checkpiotn {}".format(last_checkpoint_file))

    lstm = SimpleLSTM()
    lstm.load_data()
    targets = lstm.dataset.targets.copy()


    # lstm.plot_dataframe(features=lstm.dataset.features, targets=lstm.dataset.targets,
    #                     feature_names=lstm.dataset.feature_names,
    #                     target_names=lstm.dataset.target_names)

    # Preprocess the data by scaling, smoothing and shifting.
    lstm.preprocess_data()

    # lstm.plot_dataframe(features=lstm.dataset.features, targets=lstm.dataset.targets,
    #                     feature_names=lstm.dataset.feature_names,
    #                     target_names=lstm.dataset.target_names)

    lstm.prepare_train_test_data()
    lstm.create_model(checkpoint_path=last_checkpoint_file)

    features = lstm.dataset.features

    features = features[1000:1200]
    targets = targets[1000:1200]

    print("Feature shape here: {}".format(features.shape))
    print("Tragets shape here: {}".format(targets.shape))

    inference_data_x, inference_data_y = \
        lstm.create_supervised_data(features, targets, lstm.look_back, lstm.look_front)

    yhat = lstm.inference(inference_data=inference_data_x)

    targets = targets[lstm.look_back + 1:]
    lstm.plot_inference(yhat, ground_truth=targets)
