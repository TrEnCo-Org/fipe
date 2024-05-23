import gurobipy as gp
import pandas as pd

from ..typing import Sample
from ..ensemble import (
    Features,
    Ensemble
)


class MIP:
    model: gp.Model

    def build(self, name=""):
        self.model = gp.Model(name)

    def set_gurobi_param(self, param, value):
        self.model.setParam(param, value)


class EPS:
    eps: float

    def __init__(self, **kwargs):
        self.eps = kwargs.get("eps", 1.0)


class WeightedModel:
    weights: dict[int, float]

    def __init__(self, weights):
        self.weights = dict()
        for t, w in enumerate(weights):
            self.weights[t] = w

    @property
    def min_weight(self):
        return min(self.weights.values())


class FeatureContainer:
    features: Features

    def __init__(self, features: Features):
        self.features = features

    def transform(self, X: Sample | list[Sample]):
        if not isinstance(X, list):
            X = [X]

        df = pd.DataFrame(X, columns=self.columns)
        return df.values

    @property
    def continuous(self):
        return self.features.continuous

    @property
    def categorical(self):
        return self.features.categorical

    @property
    def binary(self):
        return self.features.binary

    @property
    def n_features(self):
        return self.features.n_features

    @property
    def categories(self):
        return self.features.categories

    @property
    def lower_bounds(self):
        return self.features.lower_bounds

    @property
    def upper_bounds(self):
        return self.features.upper_bounds

    @property
    def columns(self):
        return self.features.columns


class EnsembleContainer:
    ensemble: Ensemble

    def __init__(self, ensemble: Ensemble):
        self.ensemble = ensemble

    @property
    def n_estimators(self):
        return self.ensemble.n_estimators

    @property
    def n_classes(self):
        return self.ensemble.n_classes
