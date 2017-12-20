from datetime import timedelta, datetime

import numpy as np

from simple_lstm import Dataset


def gaussian(x: np.ndarray) -> np.ndarray:
    linspace = np.arange(0, len(x))
    center = len(x) // 2

    deviation = np.random.normal(loc=len(x) / 5, scale=len(x) / 10)
    mean = np.random.normal(loc=center, scale=len(x) / 3)

    return np.exp(-((linspace - mean) / deviation) ** 2)


class DatasetCreatorParams:
    def __init__(self, num_features: int = 9, num_targets: int = 1,
                 num_samples: int = 1000, sample_dt: timedelta = timedelta(minutes=30),
                 sample_dx: float = np.pi / 100.0, functions: tuple = (np.sin, np.cos),
                 frequency_scale: float = None, amplitude_scale: float = 1.0,
                 random_seed: int = None, randomize: bool = True):
        self.num_features = num_features
        self.num_targets = num_targets
        self.num_samples = num_samples
        self.sample_dt = sample_dt
        self.sample_dx = sample_dx

        self.functions = list(functions)
        if frequency_scale is None:
            frequency_scale = 0.25 / sample_dx
        self.frequency_scale = frequency_scale
        self.amplitude_scale = amplitude_scale

        if random_seed is not None:
            np.random.seed(random_seed)

        self.randomize = randomize


class DatasetCreator:
    def __init__(self, params: DatasetCreatorParams = DatasetCreatorParams()):

        self.params = params

    def create(self) -> Dataset:

        base_date = datetime(2000, 1, 1)
        time_linspace = np.array([base_date + i * self.params.sample_dt
                                  for i in range(self.params.num_samples)])

        start_x = 0
        end_x = self.params.sample_dx * self.params.num_samples
        x_linspace = np.linspace(start_x, end_x, self.params.num_samples)

        rand = np.random.rand
        choice = np.random.choice
        if not self.params.randomize:
            rand = lambda: 1

            def choice_(x: list):
                c = x[0]
                new_x = x[1:]
                new_x.append(c)
                x = new_x
                return c

        features = []
        for i in range(self.params.num_features):
            amplitude = self.params.amplitude_scale * rand()
            frequency = self.params.frequency_scale * rand()
            funct = choice(self.params.functions)
            feature = amplitude * funct(frequency * x_linspace)
            features.append(feature)

        features = np.atleast_2d(np.array(features).T)
        if features.shape[0] == 1:
            features = features.T

        targets = []
        for i in range(self.params.num_targets):
            target = np.zeros(features.shape[0])
            for feature in features.T:
                target += rand() * feature
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
