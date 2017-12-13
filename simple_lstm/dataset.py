import json
import os
from enum import Enum
from typing import List

import numpy as np
import pandas as pd


class ColumnType(Enum):
    TIME = 0
    FEATURE = 1
    TARGET = 2


class CsvColumn:
    def __init__(self, col_index, name, type, normalize):
        self.col_index = col_index  # type: int
        self.name = name  # type: str
        self.type = ColumnType[type.upper()]  # type: ColumnType
        self.normalize = normalize  # type: bool

    def __repr__(self):
        s = "{:>2}) {:<25} {:>7} {:^6} normalize".format(
            self.col_index, self.name, self.type.name,
            "do" if self.normalize else "do not")
        return s


CsvColumnList = List[CsvColumn]


class DatasetLoader:
    def __init__(self, csv_path: str, meta_data_path: str):

        self.csv_path = csv_path
        self.meta_data_path = meta_data_path

        self.meta_data = None  # type: CsvColumnList

        self.dataframe = None  # type: np.ndarray
        self.timestamp = None  # type: np.ndarray

        self.nan_fraction_accepted = 0.1

        self.feature_indices = None  # type: list
        self.feature_names = None  # type: list
        self.target_indices = None  # type: list
        self.target_names = None  # type: list

        self.__load()

    def __load(self):
        self.__load_meta_data()
        self.__load_csv_file()

    @property
    def csv_path(self):
        return self.__csv_path

    @csv_path.setter
    def csv_path(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError("CSV file {} does not exist!".format(csv_path))

        self.__csv_path = csv_path

    @property
    def meta_data_path(self):
        return self.__meta_data_path

    @meta_data_path.setter
    def meta_data_path(self, meta_data_path: str):
        if not os.path.exists(meta_data_path):
            raise FileNotFoundError(
                "Meta data file {} does not exist!".format(meta_data_path))

        self.__meta_data_path = meta_data_path

    @property
    def features(self) -> np.ndarray:
        return np.atleast_2d(self.dataframe[:, self.feature_indices])

    @features.setter
    def features(self, f: np.ndarray):
        assert (f.shape == self.features.shape)
        self.dataframe[:, self.feature_indices] = f

    @property
    def targets(self) -> np.ndarray:
        t = np.atleast_2d(self.dataframe[:, self.target_indices])
        if t.shape[0] == 1:
            t = t.T
        return t

    @targets.setter
    def targets(self, t: np.ndarray):
        t = np.atleast_2d(t)
        assert (t.shape == self.targets.shape)
        self.dataframe[:, self.target_indices] = t

    def __load_meta_data(self):
        with open(self.meta_data_path, 'r') as json_meta_data_file:
            json_dict = json.load(json_meta_data_file)

        column_list = []  # type: CsvColumnList
        for column_dict in json_dict["columns"]:
            column = CsvColumn(**column_dict)
            column_list.append(column)

        self.meta_data = sorted(column_list, key=lambda x: x.col_index)

    def __load_csv_file(self):
        dataframe = pd.read_csv(self.csv_path, delimiter=",", index_col=False,
                                usecols=[meta.col_index for meta in self.meta_data])

        # Drop all columns which have too many nans.
        num_rows = dataframe.shape[0]
        num_nans_accepted = int(self.nan_fraction_accepted * num_rows)
        drop_col_indices = []
        for col_index, col in enumerate(dataframe.columns):
            num_nans = np.count_nonzero(dataframe[col].isnull())
            if num_nans > num_nans_accepted:
                drop_col_indices.append(col_index)
                print("Ignoring feature {} as there are too many 'nan's".format(col))

        self.meta_data = [meta for meta in self.meta_data
                          if meta.col_index not in drop_col_indices]
        dataframe.drop(dataframe.columns[drop_col_indices], axis=1, inplace=True)

        # Drop all rows which still contain a nan.
        dataframe.dropna(inplace=True)

        # Reorder columns inside the dataframe.
        time_name = [meta.name for meta in self.meta_data
                     if meta.type == ColumnType.TIME][0]

        # Rename and reorder dataframe columns
        self.feature_names = [meta.name for meta in self.meta_data
                              if meta.type == ColumnType.FEATURE]

        self.target_names = [meta.name for meta in self.meta_data
                             if meta.type == ColumnType.TARGET]

        dataframe = dataframe[[time_name, *self.feature_names, *self.target_names]]

        self.timestamp = dataframe.values[:, 0].copy()
        dataframe.drop(labels=dataframe.columns[0], inplace=True, axis=1)

        self.feature_indices = [i for i in range(len(self.feature_names))]
        self.target_indices = [i + self.feature_indices[-1] + 1
                               for i in range(len(self.target_names))]

        # Make sure the data is all in floating point representation.
        self.dataframe = dataframe.values.astype(np.float32)
