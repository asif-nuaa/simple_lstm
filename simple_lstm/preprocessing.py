from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class __Transformer(ABC):
    def __init__(self, features_shape: tuple, targets_shape: tuple):
        super().__init__()

        # Store the shapes a protected, accessible by subclasses
        self._features_shape = features_shape
        self._targets_shape = targets_shape


    @abstractmethod
    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        pass


class DataScaler(__Transformer):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        super().__init__(features.shape, targets.shape)

        self.__feature_scaler = MinMaxScaler()
        self.__target_scaler = MinMaxScaler()

        self.__feature_scaler.fit(features)
        self.__target_scaler.fit(targets)

    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        transformed_features = self.__transform_features(features)
        transformed_targets = self.__transform_targets(targets)

        return transformed_features, transformed_targets

    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        restored_features = self.__restore_features(features)
        restored_targets = self.__restore_targets(targets)

        return restored_features, restored_targets

    def __transform_features(self, features: np.ndarray) -> np.ndarray:
        assert (features.shape[1] == self._features_shape[1])

        return self.__feature_scaler.transform(features)

    def __transform_targets(self, targets: np.ndarray) -> np.ndarray:
        assert (targets.shape[1] == self._targets_shape[1])

        return self.__target_scaler.transform(targets)

    def __restore_features(self, features: np.ndarray) -> np.ndarray:
        assert (features.shape[1] == self._features_shape[1])

        return self.__feature_scaler.inverse_transform(features)

    def __restore_targets(self, targets: np.ndarray) -> np.ndarray:
        assert (targets.shape[1] == self._targets_shape[1])

        return self.__target_scaler.inverse_transform(targets)


class DataPreprocessor(__Transformer):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        super().__init__(features_shape=features.shape, targets_shape=targets.shape)

        self.__transformers = []

    def add_transformer(self, transformer: super.__class__):
        self.__transformers.append(transformer)

    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:

        if len(self.__transformers) == 0:
            RuntimeWarning("The data preprocessor does not have a transformer applied "
                           "to it. Use 'DataPreprocessor.add(transformer)' before "
                           "transforming any data.")
            return features, targets

        for transformer in self.__transformers:
            features, targets = transformer.transform(features, targets)

        return features, targets

    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        if len(self.__transformers) == 0:
            RuntimeWarning("The data preprocessor does not have a transformer applied "
                           "to it. Use 'DataPreprocessor.add(transformer)' before "
                           "restoring any data.")
            return features, targets

        for transformer in reversed(self.__transformers):
            features, targets = transformer.restore(features, targets)

        return features, targets

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        empty_features = np.ones(shape=(1, self._features_shape[1]))
        _, targets = self.transform(empty_features, targets)
        return targets

    def restore_targets(self, targets: np.ndarray) -> np.ndarray:
        empty_features = np.ones(shape=(1, self._features_shape[1]))
        _, targets = self.restore(empty_features, targets)
        return targets
