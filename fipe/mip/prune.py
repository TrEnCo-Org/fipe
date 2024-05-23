from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from sklearn.ensemble._base import BaseEnsemble
from sklearn.ensemble._iforest import IsolationForest

from .abc import (
    MIP,
    EPS,
    WeightedModel,
    FeatureContainer,
    EnsembleContainer
)
from .oracle import Oracle
from ..typing import Sample
from ..ensemble import (
    Features,
    Ensemble,
    IsolationEnsemble,
    predict,
    predict_single_proba
)

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BasePruner(ABC):
    @abstractmethod
    def prune(self):
        raise NotImplementedError("The prune method must be implemented.")


class Pruner(
    BasePruner,
    MIP,
    EPS,
    WeightedModel
):
    base: BaseEnsemble
    activated_weights: dict[int, gp.Var]

    def __init__(
        self,
        base: BaseEnsemble,
        weights,
        **kwargs
    ):
        MIP.__init__(self)
        EPS.__init__(self, **kwargs)
        WeightedModel.__init__(self, weights)
        self.base = base
        self.activated_weights = dict()

    def build(self):
        MIP.build(self, name="FIPE")
        self._add_vars()
        self._add_objective()

    def prune(self):
        self.model.optimize()

    def add_sample_constrs(self, X):
        m = len(self.base)
        w = np.array([
            self.weights[t]
            for t in range(m)
        ])
        y = predict(self.base, X, w)
        p = predict_single_proba(self.base, X)
        n = X.shape[0]
        eps = self.eps
        mw = self.min_weight
        for i in range(n):
            for c in range(self.n_classes):
                if c == y[i]:
                    continue
                rhs = 0.0 if y[i] < c else eps * mw
                lhs = gp.quicksum(
                    self.activated_weights[t]
                    * (p[i, t, y[i]] - p[i, t, c])
                    for t in range(m)
                )
                self.model.addConstr(lhs >= rhs)

    @property
    def pruned_weights(self) -> dict[int, float]:
        if self.model.SolCount == 0:
            logger.warning("No pruning solution found.")
            return {
                t: 1.0
                for t in self.activated_weights
            }
        return {
            t: self.activated_weights[t].X
            for t in self.activated_weights
        }

    @property
    def n_estimators(self):
        return len(self.base)

    @property
    def n_classes(self) -> int:
        return self.base[0].n_classes_

    def _add_vars(self):
        for t in range(self.n_estimators):
            self.activated_weights[t] = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"activated_weights_{t}"
            )

    def _add_objective(self):
        self.model.setObjective(
            gp.quicksum(
                self.activated_weights[t]
                for t in self.activated_weights
            ),
            GRB.MINIMIZE
        )


class FullPruner(
    Pruner,
    FeatureContainer,
    EnsembleContainer,
):
    oracle: Oracle
    n_oracle_calls: int
    max_oracle_calls: int
    counters: list[list[Sample]]
    history: list[dict[int, float]]

    def __init__(
        self,
        base: BaseEnsemble,
        weights,
        features: Features,
        isolation_forest: IsolationForest | None = None,
        **kwargs
    ):
        Pruner.__init__(self, base, weights, **kwargs)
        FeatureContainer.__init__(self, features)
        ensemble = Ensemble(base, features)
        EnsembleContainer.__init__(self, ensemble)
        if isolation_forest is None:
            isolation_ensemble = None
        else:
            isolation_ensemble = IsolationEnsemble(
                isolation_forest,
                features
            )
        self.oracle = Oracle(
            features=features,
            ensemble=ensemble,
            weights=weights,
            isolation_ensemble=isolation_ensemble,
            **kwargs
        )
        self.counters = []
        self.history = []
        self.max_oracle_calls = kwargs.get("max_oracle_calls", 100)

    def build(self):
        Pruner.build(self)
        self.oracle.build()
        self.n_oracle_calls = 0

    def prune(self):
        while self.n_oracle_calls < self.max_oracle_calls:
            self.model.optimize()
            if self.model.SolCount == 0:
                msg = "No pruning solution found..."
                logger.warning(msg)
                break

            self._save_pruned_weights()
            X = self._call_oracle()
            if len(X) == 0:
                msg = "No more samples to separate. The pruning converged."
                logger.info(msg)
                break
            else:
                self._save_counters(X)
                X = self.transform(X)
                self.add_sample_constrs(X)

    def _save_pruned_weights(self):
        pruned_weights = deepcopy(self.pruned_weights)
        self.history.append(pruned_weights)

    def _save_counters(self, counters):
        self.counters.append(counters)

    def _call_oracle(self):
        pruned_weights = self.pruned_weights
        self.n_oracle_calls += 1
        return list(self.oracle.separate(pruned_weights))
