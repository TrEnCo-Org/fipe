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
    _base: BaseEnsemble
    _activated_vars: gp.tupledict[int, gp.Var]
    _weights_vars: gp.tupledict[int, gp.Var]
    _continuous: bool

    def __init__(
        self,
        base: BaseEnsemble,
        weights,
        **kwargs
    ):
        MIP.__init__(self)
        EPS.__init__(self, **kwargs)
        WeightedModel.__init__(self, weights)
        self._base = base
        self._activated_vars = gp.tupledict()
        self._weights_vars = gp.tupledict()
        self._continuous = kwargs.get("continuous", False)

    def build(self):
        self._add_vars()
        self._add_objective()

    def prune(self):
        self.optimize()

    def add_sample_constrs(self, X):
        m = len(self._base)
        w = np.array([self._weights[t] for t in range(m)])
        y = predict(self._base, X, w)
        p = predict_single_proba(self._base, X)
        n = X.shape[0]
        eps = self._eps
        mw = self.min_weight
        for i in range(n):
            for c in range(self.n_classes):
                if c == y[i]:
                    continue
                rhs = 0.0 if y[i] < c else eps * mw
                lhs = gp.quicksum(
                    self._weights_vars[t]
                    * (p[i, t, y[i]] - p[i, t, c])
                    for t in range(m)
                )
                self.addConstr(lhs >= rhs)

    @property
    def weights(self) -> dict[int, float]:
        if self.SolCount == 0:
            logger.warning("No pruning solution found.")
            return {
                t: self._weights[t]
                for t in range(self.n_estimators)
            }
        return {
            t: self._weights_vars[t].X
            for t in range(self.n_estimators)
        }

    @property
    def n_activated(self) -> int:
        if self.SolCount == 0:
            logger.warning("No pruning solution found.")
            return self.n_estimators
        activated = np.array([
            self._activated_vars[t].X > 0.5
            for t in range(self.n_estimators)
        ])
        return int(activated.sum())

    @property
    def n_estimators(self):
        return len(self._base)

    @property
    def n_classes(self) -> int:
        return self._base[0].n_classes_

    def _add_vars(self):
        if self._continuous:
            self._add_continuous_vars()
        else:
            self._add_binary_vars()

    def _add_binary_vars(self):
        for t in range(self.n_estimators):
            self._activated_vars[t] = self.addVar(
                vtype=GRB.BINARY,
                name=f"activated_weights_{t}"
            )
            self._weights_vars[t] = self._activated_vars[t]

    def _add_continuous_vars(self):
        for t in range(self.n_estimators):
            self._activated_vars[t] = self.addVar(
                vtype=GRB.BINARY,
                name=f"activated_weights_{t}"
            )
            self._weights_vars[t] = self.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"weight_{t}"
            )
            self.addConstr(
                self._weights_vars[t]
                <=
                self._weights[t] *
                self._activated_vars[t]
            )

    def _add_objective(self):
        self.setObjective(
            gp.quicksum(
                self._activated_vars[t]
                for t in self._activated_vars
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
    _counters: list[list[Sample]]
    _history: list[dict[int, float]]

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
        self._counters = []
        self._history = []
        self.max_oracle_calls = kwargs.get("max_oracle_calls", 100)

    def build(self):
        Pruner.build(self)
        self.oracle.build()
        self.n_oracle_calls = 0

    def prune(self):
        while self.n_oracle_calls < self.max_oracle_calls:
            self.optimize()
            if self.SolCount == 0:
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
        pruned_weights = deepcopy(self.weights)
        self._history.append(pruned_weights)

    def _save_counters(self, counters):
        self._counters.append(counters)

    def _call_oracle(self):
        weights = self.weights
        self.n_oracle_calls += 1
        return list(self.oracle(weights))
