from datetime import timedelta, datetime

import numpy as np

from simple_lstm import Dataset


class DatasetCreatorParams:
    def __init__(self, num_features: int = 9, num_targets: int = 1,
                 num_samples: int = 10000, sample_dt: timedelta = timedelta(minutes=30),
                 functions: set = {np.sin, np.cos},
                 frequency_scale: float = 1.0, amplitude_scale: float = 1.0,
                 random_seed: int = None):
        self.num_features = num_features
        self.num_targets = num_targets
        self.num_samples = num_samples
        self.sample_dt = sample_dt

        self.functions = list(functions)
        self.frequency_scale = frequency_scale
        self.amplitude_scale = amplitude_scale

        if random_seed is not None:
            np.random.seed(random_seed)


class DatasetCreator:
    def __init__(self, params: DatasetCreatorParams = DatasetCreatorParams()):

        self.params = params

    def create(self) -> Dataset:

        base_date = datetime(2000, 1, 1)
        time_linspace = np.array([base_date + i * self.params.sample_dt
                                  for i in range(self.params.num_samples)])
        x_linspace = np.linspace(0, 100 * np.pi, self.params.num_samples)


        features = []
        for i in range(self.params.num_features):
            amplitude = self.params.amplitude_scale * np.random.rand()
            frequency = self.params.frequency_scale * np.random.rand()
            funct = np.random.choice(self.params.functions)
            feature = amplitude * funct(frequency * x_linspace + np.random.rand())
            features.append(feature)

        features = np.atleast_2d(np.array(features).T)
        if features.shape[0] == 1:
            features = features.T

        targets = []
        for i in range(self.params.num_targets):
            target = np.zeros(features.shape[0])
            for feature in features.T:
                target += np.random.rand() * feature
            targets.append(target)

        targets = np.atleast_2d(np.array(targets).T)
        if targets.shape[0] == 1:
            targets = targets.T

        dataframe = np.concatenate((features, targets), axis=1)
        feature_indices = np.arange(self.params.num_features)
        feature_names = ["f_{}".format(i + 1) for i in range(self.params.num_features)]
        target_indices = np.arange(self.params.num_features,
                                   self.params.num_features + self.params.num_targets)
        target_names = ["t_{}".format(i + 1) for i in range(self.params.num_targets)]
        dataset = Dataset(timestamp=time_linspace, dataframe=dataframe,
                          feature_indices=feature_indices,
                          feature_names=feature_names,
                          target_indices=target_indices, target_names=target_names)

        return dataset
