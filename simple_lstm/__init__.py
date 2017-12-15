import os

from .dataset import Dataset
from .creator import DatasetCreatorParams, DatasetCreator, gaussian
from .loader import DatasetLoader
from .saver import  get_saver_callback
from .preprocessing import DataPreprocessor, DataScaler


class __Settings:
    def __init__(self):
        self.simple_lstm_root = os.path.dirname(os.path.realpath(__file__))
        self.dataset_root = os.path.join(
            os.path.join(self.simple_lstm_root, os.pardir), "dataset")
        self.checkpoint_root = os.path.join(
            os.path.join(self.simple_lstm_root, os.pardir), "checkpoint")

        self.__check_directories()

    def __check_directories(self):
        for attr in dir(self):
            if "root" in attr:
                if not os.path.exists(self.__dict__[attr]):
                    print("Creating {} directory".format(self.__dict__[attr]))
                    os.makedirs(self.__dict__[attr], exist_ok=True)

Settings = __Settings()
