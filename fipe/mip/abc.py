from typing import Any
import gurobipy as gp
import pandas as pd

from ..typing import Sample
from ..ensemble import (
    Features,
    Ensemble
)


class MIP(gp.Model):
    def __init__(
        self,
        name: str = "",
        env: gp.Env | None = None
    ):
        gp.Model.__init__(self, name, env)

    def set_param(self, param, value):
        gp.Model.setParam(self, param, value)

    def __setattr__(self, name: str, value: Any) -> None:
        return object.__setattr__(self, name, value)


class EPS:
    _eps: float

    def __init__(self, **kwargs):
        self._eps = kwargs.get("eps", 1.0)


class WeightedModel:
    _weights: dict[int, float]

    def __init__(self, weights):
        self._weights = dict[int, float]()
        for t, w in enumerate(weights):
            self._weights[t] = w

    @property
    def min_weight(self):
        return min(self._weights.values())


class FeatureContainer:
    _features: Features

    def __init__(self, features: Features):
        self._features = features

    def transform(self, X: Sample | list[Sample]):
        if not isinstance(X, list):
            X = [X]

        df = pd.DataFrame(X, columns=self.columns)
        return df.values

    @property
    def continuous(self):
        return self._features.continuous

    @property
    def categorical(self):
        return self._features.categorical

    @property
    def binary(self):
        return self._features.binary

    @property
    def n_features(self):
        return self._features.n_features

    @property
    def categories(self):
        return self._features.categories

    @property
    def lower_bounds(self):
        return self._features.lower_bounds

    @property
    def upper_bounds(self):
        return self._features.upper_bounds

    @property
    def columns(self):
        return self._features.columns


class EnsembleContainer:
    _ensemble: Ensemble

    def __init__(self, ensemble: Ensemble):
        self._ensemble = ensemble

    @property
    def n_estimators(self):
        return self._ensemble.n_estimators

    @property
    def n_classes(self):
        return self._ensemble.n_classes
