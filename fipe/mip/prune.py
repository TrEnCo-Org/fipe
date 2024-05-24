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
        msg = "The prune method must be implemented."
        raise NotImplementedError(msg)


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

    _sample_constrs: gp.tupledict[tuple[int, int], gp.Constr]
    _n_samples: int

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
        self._n_samples = 0

    def add_samples(self, X):
        w = np.array([
            self._weights[t]
            for t in range(self.n_estimators)
        ])
        y = predict(self._base, X, w)
        p = predict_single_proba(self._base, X)
        n = X.shape[0]
        for i in range(n):
            self._add_sample_constrs(p[i], y[i])

    def prune(self):
        if self._n_samples == 0:
            msg = "No samples was added to the pruner."
            raise ValueError(msg)
        self.optimize()

    @property
    def n_estimators(self):
        return len(self._base)

    @property
    def n_classes(self) -> int:
        return self._base[0].n_classes_

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
                self._weights[t]
                * self._activated_vars[t]
            )

    def _add_sample_constrs(self, p, y: int):
        for c in range(self.n_classes):
            if c == y:
                continue
            self._add_sample_constr(p, y, c)
        self._n_samples += 1

    def _add_sample_constr(self, p, y: int, c: int):
        i = self._n_samples
        mw = self.min_weight
        self._sample_constrs[i, c] = self.addConstr(
            gp.quicksum(
                self._weights_vars[t]
                * (p[t, y] - p[t, c])
                for t in range(self.n_estimators)
            ) >= (mw * self.get_cutoff(y, c)),
            name=f"sample_{i}_{c}"
        )

    def _add_objective(self):
        self.setObjective(
            gp.quicksum(
                self._activated_vars[t]
                for t in range(self.n_estimators)
            ),
            GRB.MINIMIZE
        )


class FullPruner(
    Pruner,
    FeatureContainer,
    EnsembleContainer,
):
    oracle: Oracle
    _n_oracle_calls: int
    _max_oracle_calls: int
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
        self._max_oracle_calls = kwargs.get("max_oracle_calls", 100)

    def build(self):
        Pruner.build(self)
        self.oracle.build()
        self._n_oracle_calls = 0

    def prune(self):
        while self._n_oracle_calls < self._max_oracle_calls:
            self.optimize()
            if self.SolCount == 0:
                msg = "No pruning solution found..."
                logger.warning(msg)
                break

            self._save_weights()
            X = self._call_oracle()
            if len(X) == 0:
                msg = "No more samples to separate. The pruning converged."
                logger.info(msg)
                break
            else:
                self._save_counters(X)
                X = self.transform(X)
                self.add_samples(X)

    @property
    def n_oracle_calls(self) -> int:
        return self._n_oracle_calls

    def _save_weights(self):
        weights = deepcopy(self.weights)
        self._history.append(weights)

    def _save_counters(self, counters: list[Sample]):
        self._counters.append(counters)

    def _call_oracle(self):
        weights = self.weights
        self._n_oracle_calls += 1
        return list(self.oracle(weights))
