import os

from keras.callbacks import ModelCheckpoint


def get_saver_callback(checkpoint_dir: str, status_str : str = "") -> ModelCheckpoint:
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError("Checkpoint folder {} does not exist.".format(checkpoint_dir))

    checkpoint_file_name = "epoch-{epoch:03d}_val-loss-{val_loss:.4f}.hdf5"
    checkpoint_file_name = status_str + "_" + checkpoint_file_name
    checkpoint_file_name = os.path.join(checkpoint_dir, checkpoint_file_name)

    return ModelCheckpoint(filepath=checkpoint_file_name, period=1, save_best_only=False)
