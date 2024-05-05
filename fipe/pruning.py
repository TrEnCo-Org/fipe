from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

import gurobipy as gp
from gurobipy import GRB

from ._predict import (
    predict_single_proba,
    predict
)
from .encoding import FeatureEncoder
from .tree import TreeEnsemble
from .separation import FIPEOracle

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BasePruner(ABC):
    @abstractmethod
    def prune(self):
        raise NotImplementedError("The prune method must be implemented.")


class FIPEPruner(BasePruner):
    gurobi_model: gp.Model
    active_vars: gp.tupledict[int, gp.Var]
    eps: float

    def __init__(self, E, w, **kwargs):
        self.ensemble_model = E
        self.weights = np.array(w)
        self.eps = kwargs.get("eps", 1.0)

    def build(self):
        gurobi_model = gp.Model("FIPE")
        m = len(self.ensemble_model)

        self.active_vars = gurobi_model.addVars(
            m, vtype=GRB.BINARY,
            name="active_tree"
        )

        # Number of estimators in the ensemble.
        m = len(self.ensemble_model)
        # Add the objective function.
        # We want to minimize the number
        # of active trees: \sum_{e=1}^{m} u_e
        gurobi_model.setObjective(
            gp.quicksum(
                self.active_vars[e]
                for e in range(m)
            ),
            GRB.MINIMIZE
        )
        self.gurobi_model = gurobi_model

    def add_constraints(self, X):
        p = predict_single_proba(self.ensemble_model, X)
        y = predict(self.ensemble_model, X, self.weights)

        n = X.shape[0]
        m = len(self.ensemble_model)
        k = p.shape[-1]
        wm = self.weights.min()
        # Add the constraint that the predicted class
        # should be the same for the subset of active trees.
        for i in range(n):
            for j in range(k):
                if j == y[i]:
                    continue
                rhs = (0.0 if y[i] < j else self.eps * wm)
                cons = self.gurobi_model.addConstr(
                    gp.quicksum(
                        self.weights[e]
                        * (p[i, e, y[i]] - p[i, e, j])
                        * self.active_vars[e]
                        for e in range(m)
                    ) >= rhs,
                    name=f"sample_{i}_class_{j}"
                )
                cons.Lazy = 1

    def prune(self):
        self.gurobi_model.optimize()

    def set_gurobi_parameter(self, param, value):
        self.gurobi_model.setParam(param, value)

    @property
    def active(self):
        m = len(self.ensemble_model)
        if self.gurobi_model.SolCount == 0:
            msg = ("When solving the FIPE problem,"
                   " no solution was found.")
            logger.warning(msg)
            return np.zeros(m, dtype=bool)

        u = self.active_vars
        v = [int(u[e].X >= 0.5) for e in range(m)]
        return np.array(v)


class FIPEPrunerFull(BasePruner):
    pruner: FIPEPruner
    oracle: FIPEOracle
    max_iter: int

    def __init__(
        self,
        ensemble_model,
        weights,
        feature_encoder: FeatureEncoder,
        **kwargs
    ):
        tree_ensemble = TreeEnsemble(ensemble_model, feature_encoder, **kwargs)
        self.pruner = FIPEPruner(ensemble_model, weights, **kwargs)
        self.oracle = FIPEOracle(
            feature_encoder, tree_ensemble, weights, **kwargs)
        self.max_iter = kwargs.get("max_iter", 100)

    def build(self):
        self.pruner.build()
        self.oracle.build()

    def add_points(self, X):
        self.pruner.add_constraints(X)

    def set_pruner_gurobi_parameter(self, param, value):
        self.pruner.set_gurobi_parameter(param, value)

    def set_oracle_gurobi_parameter(self, param, value):
        self.oracle.set_gurobi_parameter(param, value)

    def prune(self):
        self.pruner.prune()
        active = deepcopy(self.pruner.active)

        it = 0
        while True:
            X = self.oracle.separate(active)
            found_counterfactual = (len(X) > 0)

            if found_counterfactual:
                self.pruner.add_constraints(X)
                self.pruner.prune()

            if np.isclose(active, self.pruner.active).all():
                break

            active = deepcopy(self.pruner.active)
            it += 1
            if it >= self.max_iter:
                msg = "Maximum number of iterations reached."
                logger.info(msg)
                break
