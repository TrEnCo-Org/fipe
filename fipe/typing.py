from enum import Enum

numeric = int | float
Sample = dict[str, numeric]


class FeatureType(Enum):
    CATEGORICAL = 1
    DISCRETE = 2
    CONTINUOUS = 3
    BINARY = 4
