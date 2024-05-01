from enum import Enum

numeric = int|float

class FeatureType(Enum):
    CATEGORICAL = 1
    NUMERICAL = 2
    BINARY = 3
    