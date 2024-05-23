import numpy as np

from ..typing import numeric

from .ensemble import Ensemble
from .features import Features

import warnings


class EnsembleParser:
    levels: dict[str, list[numeric]]
    tol: float

    def __init__(self, **kwargs):
        self.levels = dict()
        self.tol = kwargs.get("tol", 1e-4)

    def parse_levels(
        self,
        ensembles: list[Ensemble],
        features: Features
    ):
        for feature in features.continuous:
            levels = set()
            levels.add(features.lower_bounds[feature])
            levels.add(features.upper_bounds[feature])
            for ensemble in ensembles:
                levels |= self.get_levels(feature, ensemble)
            levels = list(sorted(levels))
            if np.diff(levels).min() < self.tol:
                msg = (f"The levels of the feature {feature}"
                       " are too close to each other.")
                warnings.warn(msg)
            self.levels[feature] = levels

    def get_levels(
        self,
        feature: str,
        ensemble: Ensemble
    ):
        levels = set()
        for tree in ensemble:
            for n in tree.nodes_split_on(feature):
                levels.add(tree.threshold[n])
        return levels
