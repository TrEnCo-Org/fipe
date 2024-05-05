from enum import Enum

numeric = int | float


class FeatureType(Enum):
    CATEGORICAL = 1
    DISCRETE = 2
    CONTINUOUS = 3
    BINARY = 4
