from abc import ABC, abstractmethod

import numpy as np

import gurobipy as gp
from gurobipy import GRB

import logging

from ._predict import (
    predict_single_proba,
    predict
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BasePruner(ABC):
    @abstractmethod
    def prune(self):
        pass

class FIPEPruner:
    """
    FIPE: Functionally Identical Pruning for Ensemble models.
    
    Parameters:
    -----------
    E: list
        List of estimators in the ensemble.
    w: list
        List of weights for each estimator in E.
        
    Attributes:
    -----------
    E: list
        List of estimators in the ensemble.
    w: list
        List of weights for each estimator in E.
    gurobi_model: gp.Model
        Gurobi model for the FIPE problem.
    active_tree_vars: gp.tupledict[int, gp.Var]
        Binary variables indicating whether the tree
        corresponding to the estimator is active.
    """
    gurobi_model: gp.Model
    active_vars: gp.tupledict[int, gp.Var]
    
    def __init__(self, E, w):
        self.E = E
        self.w = w

    def build(self):
        logger.debug("Building the gurobi model for FIPE.")
        gurobi_model = gp.Model("FIPE")

        self.active_vars = gurobi_model.addVars(
            len(self.E),
            vtype=GRB.BINARY,
            name="active_tree"
        )
        logger.debug("Added binary variables for active classifiers.")

        # Number of estimators in the ensemble.
        m = len(self.E)
        # Add the objective function.
        # We want to minimize the number
        # of active trees: \sum_{e=1}^{m} u_e
        gurobi_model.set_objective(
            gp.quicksum(
                self.active_vars[e]
                for e in range(m)
            ),
            GRB.MINIMIZE
        )
        logger.debug("Added the objective function.")

        # Add the constraint that at least one
        # tree should be active.
        # \sum_{e=1}^{m} u_e >= 1
        gurobi_model.addConstr(
            gp.quicksum(
                self.active_vars[e]
                for e in range(m)
            ) >= 1,
            name="at_least_one_active"
        )
        logger.debug("Added the constraint that at least one tree should be active.")
        self.gurobi_model = gurobi_model
        

    def add_constraints(self, X):
        p = predict_single_proba(self.E, X)
        y = predict(self.E, X, self.w)
        
        n = X.shape[0]
        m = len(self.E)
        k = p.shape[-1]
        
        # Add the constraint that the predicted class
        # should be the same for the subset of active trees.
        for i in range(n):
            for j in range(k):
                logger.debug(f"Adding constraint for class {j} and sample {i}.")
                # \sum_{e=1}^{m} u_e * p[i, e, y[i]] >= \sum_{e=1}^{m} u_e * p[i, e, j]
                cons = self.gurobi_model.addConstr(
                    gp.quicksum(
                        self.w[e]
                        *(p[i, e, y[i]] - p[i, e, j])
                        *self.active_vars[e]
                        for e in range(m)
                    ) >= 0.0,
                    name=f"sample_{i}_class_{j}"
                )
                cons.Lazy = 1

    def prune(self):
        self.gurobi_model.optimize()

    def set_gurobi_parameter(self, param, value):
        self.gurobi_model.setParam(param, value)

    @property
    def active(self):
        if self.gurobi_model.SolCount == 0:
            logger.warning("When solving the FIPE problem, no solution was found.")
            return []

        m = len(self.E)
        u = self.active_vars
        v = [u[e].X >= 0.5 for e in range(m)]
        return np.array(v)