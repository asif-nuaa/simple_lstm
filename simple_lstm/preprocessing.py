from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class AbstractTransformer(ABC):
    def __init__(self):
        super().__init__()

        # Store the shapes a protected, accessible by subclasses
        self._features_shape = None  # type: tuple
        self._targets_shape = None  # type: tuple

    @abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        pass

    @abstractmethod
    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        pass


class AbsTransformer(AbstractTransformer):
    def __init__(self):
        super().__init__()

        self.__feature_signs = None  # type: np.ndarray
        self.__target_signs = None  # type: np.ndarray

    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        if self.__feature_signs.shape != features.shape \
                or self.__target_signs.shape != targets.shape:
            raise RuntimeError("You are restoring data with different shape than the "
                               "one which has been transformed!")

        targets = targets * self.__target_signs
        features = features * self.__feature_signs
        return features, targets

    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        self.__feature_signs = np.sign(features)
        self.__target_signs = np.sign(targets)

        targets = np.abs(targets)
        features = np.abs(features)
        return features, targets

    def fit(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        self._features_shape = features.shape
        self._targets_shape = features.shape

        return self.transform(features, targets)


class ShiftTransformer(AbstractTransformer):
    def __init__(self, shift: float = 0):
        super().__init__()

        self.__shift = shift

    def restore(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        features -= self.__shift
        targets -= self.__shift
        return features, targets

    def transform(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        targets += self.__shift
        features += self.__shift
        return features, targets

    def fit(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        self._features_shape = features.shape
        self._targets_shape = features.shape

        return self.transform(features, targets)


class DataScaler(AbstractTransformer):
    def __init__(self):
        super().__init__()

        self.__feature_scaler = MinMaxScaler()
        self.__target_scaler = MinMaxScaler()

    def fit(self, features: np.ndarray, targets: np.ndarray) -> tuple:
        self._features_shape = features.shape
        self._targets_shape = targets.shape

        self.__feature_scaler.fit(features)
        self.__target_scaler.fit(targets)

        return self.transform(features, targets)

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


class DataPreprocessor(AbstractTransformer):
    def __init__(self, transformers=None):
        super().__init__()

        if isinstance(transformers, list):
            for transformer in transformers:
                if not isinstance(transformer, AbstractTransformer):
                    raise RuntimeError(
                        "Transformer passed to DataPreprocessor does not "
                        "inherit from {}".format(AbstractTransformer))
            self.__transformers = transformers
        elif transformers is None:
            self.__transformers = []
        else:
            raise RuntimeError("Parameter 'transformers' passed to DataPreprocessor is "
                               "not a valid type. Accepted types: list[AbstractTransformer]")

    def add_transformer(self, transformer: AbstractTransformer):
        self.__transformers.append(transformer)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:

        self._features_shape = features.shape
        self._targets_shape = features.shape

        if len(self.__transformers) == 0:
            RuntimeWarning("The data preprocessor does not have a transformer applied "
                           "to it. Use 'DataPreprocessor.add(transformer)' before "
                           "fitting any data.")
            return

        f, t = features.copy(), targets.copy()
        for transformer in self.__transformers:
            f, t = transformer.fit(f, t)

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
