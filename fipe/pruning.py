from abc import ABC, abstractmethod

import numpy as np

import gurobipy as gp
from gurobipy import GRB

from ._predict import (
    predict_single_proba,
    predict
)
from .encoding import FeatureEncoder

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class BasePruner(ABC):
    @abstractmethod
    def prune(self):
        pass

class FIPEPruner:
    """
    FIPE: Functionally Identical Pruning for Ensemble models.
    
    This class implements the FIPE algorithm for pruning ensemble models on finite subset of data.
    
    Parameters
    ----------
    E : list
        List of ensemble models. Each model should have a `predict_proba` method that returns the probability of each class.
    w : list of float
        Weights of the ensemble models.

    Attributes
    ----------
    ensemble_model :
        The ensemble model.
    weights :
        The weights of the ensemble models.
    gurobi_model :
        The gurobi model for the FIPE problem.
    active_vars :
        The binary variables for the active classifiers.
    """
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
                        *(p[i, e, y[i]] - p[i, e, j])
                        *self.active_vars[e]
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
            logger.warning("When solving the FIPE problem, no solution was found.")
            return np.zeros(m, dtype=bool)

        u = self.active_vars
        v = [u[e].X >= 0.5 for e in range(m)]
        return np.array(v)   

class FIPEOracle:
    feature_encoder: FeatureEncoder
    pass

class FIPEPrunerFull:
    base_pruner: FIPEPruner
    oracle: FIPEOracle
    pass