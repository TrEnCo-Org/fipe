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
        for f in features.continuous:
            levels = set()
            levels.add(features.lower_bounds[f])
            levels.add(features.upper_bounds[f])
            for ensemble in ensembles:
                levels |= self.get_levels(f, ensemble)
            levels = list(sorted(levels))
            if np.diff(levels).min() < self.tol:
                msg = (f"The levels of the feature {f}"
                       " are too close to each other.")
                warnings.warn(msg)
            self.levels[f] = levels

    def get_levels(self, f: str, ensemble: Ensemble):
        levels = set()
        for tree in ensemble:
            for n in tree.nodes_split_on(f):
                levels.add(tree.threshold[n])
        return levels
