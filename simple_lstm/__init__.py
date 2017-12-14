import os

from .dataset import Dataset
from .creator import DatasetCreatorParams, DatasetCreator
from .loader import DatasetLoader


class __Settings:
    def __init__(self):
        self.simple_lstm_root = os.path.dirname(os.path.realpath(__file__))
        self.dataset_root = os.path.join(os.path.join(self.simple_lstm_root,
                                                      os.pardir), "dataset")


Settings = __Settings()
