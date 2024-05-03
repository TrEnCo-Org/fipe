from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from ._predict import (
    predict_proba,
    predict_single_proba,
    predict
)
from .encoding import FeatureEncoder
from .tree import TreeEnsemble

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class BasePruner(ABC):
    @abstractmethod
    def prune(self):
        pass

class FIPEPruner:
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
    tree_ensemble: TreeEnsemble
    
    gurobi_model: gp.Model

    # Trees:
    flow_vars: gp.tupledict[tuple[int, int], gp.Var]
    branch_vars: gp.tupledict[tuple[int, int], gp.Var]
    root_constraints: gp.tupledict[int, gp.Constr]
    flow_constraints: gp.tupledict[tuple[int, int], gp.Constr]
    branch_to_left_constraints: gp.tupledict[tuple[int, int], gp.Constr]
    branch_to_right_constraints: gp.tupledict[tuple[int, int], gp.Constr]
    
    # Features:
    binary_vars: gp.tupledict[str, gp.Var]
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var]
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var]
    categorical_vars: gp.tupledict[str, gp.Var]
    binary_left_constraints: gp.tupledict[tuple[str, int, int], gp.Constr]
    binary_right_constraints: gp.tupledict[tuple[str, int, int], gp.Constr]
    discrete_left_constraints: gp.tupledict[tuple[str, int, int, int], gp.Constr]
    discrete_right_constraints: gp.tupledict[tuple[str, int, int, int], gp.Constr]
    discrete_logical_constraints: gp.tupledict[tuple[str, int], gp.Constr]
    continuous_left_constraints: gp.tupledict[tuple[str, int, int, int], gp.Constr]
    continuous_right_constraints: gp.tupledict[tuple[str, int, int, int], gp.Constr]
    continuous_logical_constraints: gp.tupledict[tuple[str, int], gp.Constr]
    categorical_left_constraints: gp.tupledict[tuple[str, int, int], gp.Constr]
    categorical_right_constraints: gp.tupledict[tuple[str, int, int], gp.Constr]
    categorical_logical_constraints: gp.tupledict[str, gp.Constr]
    
    # Probabilities:
    prob_vars: gp.tupledict[int, gp.Var]
    prob_constraints: gp.tupledict[int, gp.Constr]

    prune_prob_vars: gp.tupledict[int, gp.Var]
    prune_prob_constraints: gp.tupledict[int, gp.Constr]
    majority_class_constraints: gp.tupledict[int, gp.Constr]
    
    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        tree_ensemble: TreeEnsemble,
        w,
        **kwargs
    ):
        self.feature_encoder = feature_encoder
        self.tree_ensemble = tree_ensemble
        self.weights = np.array(w)
        self.eps = kwargs.get("eps", 1.0)

    def build_trees(self):
        gurobi_model = self.gurobi_model
        tree_ensemble = self.tree_ensemble
        for t, tree in enumerate(tree_ensemble):
            for n in tree:
                var = gurobi_model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    name=f"flow_{t}_{n}"
                )
                self.flow_vars[t, n] = var

            for d in range(tree.max_depth):
                var = gurobi_model.addVar(
                    vtype=GRB.BINARY,
                    name=f"branch_{t}_{d}"
                )
                self.branch_vars[t, d] = var
        
        for t, tree in enumerate(tree_ensemble):
            root = tree.root
            cons = gurobi_model.addConstr(
                self.flow_vars[t, root] == 1.0,
                name=f"root_{t}"
            )
            self.root_constraints[t] = cons

            for n in tree.internal_nodes:
                l = tree.left[n]
                r = tree.right[n]
                cons = gurobi_model.addConstr(
                    self.flow_vars[t, n] == 
                    self.flow_vars[t, l] + self.flow_vars[t, r],
                    name=f"flow_{t}_{n}"
                )
                self.flow_constraints[t, n] = cons

                d = tree.node_depth[n]
                cons = gurobi_model.addConstr(
                    self.branch_vars[t, d] >=
                    self.flow_vars[t, l],
                    name=f"branch_to_left_{t}_{n}"
                )
                self.branch_to_left_constraints[t, n] = cons

                cons = gurobi_model.addConstr(
                    1-self.branch_vars[t, d] >=
                    self.flow_vars[t, r],
                    name=f"branch_to_right_{t}_{n}"
                )
                self.branch_to_right_constraints[t, n] = cons

    def build_features(self):
        self.add_binary_vars()
        self.add_discrete_vars()
        self.add_continuous_vars()
        self.add_categorical_vars()
        
        self.add_binary_constraints()
        self.add_discrete_constraints()
        self.add_continuous_constraints()
        self.add_categorical_constraints()

    def add_binary_vars(self):
        gurobi_model = self.gurobi_model
        feature_encoder = self.feature_encoder
        for f in feature_encoder.binary_features:
            var = gurobi_model.addVar(
                vtype=GRB.BINARY,
                name=f"binary_{f}"
            )
            self.binary_vars[f] = var

    def add_discrete_vars(self):
        gurobi_model = self.gurobi_model
        feature_encoder = self.feature_encoder
        for f in feature_encoder.discrete_features:
            n = len(feature_encoder.values[f])
            for v in range(n):
                var = gurobi_model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    name=f"discrete_{f}_{v}"
                )
                self.discrete_vars[f, v] = var

    def add_continuous_vars(self):
        gurobi_model = self.gurobi_model
        tree_ensemble = self.tree_ensemble
        feature_encoder = self.feature_encoder
        for f in feature_encoder.continuous_features:
            n = len(tree_ensemble.numerical_levels[f])
            for i in range(n-1):
                var = gurobi_model.addVar(
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    name=f"continuous_{f}_{i}"
                )
                self.continuous_vars[f, i] = var

    def add_categorical_vars(self):
        gurobi_model = self.gurobi_model
        feature_encoder = self.feature_encoder
        for f in feature_encoder.categorical_features:
            for c in feature_encoder.categories[f]:
                var = gurobi_model.addVar(
                    vtype=GRB.BINARY,
                    name=f"categorical_{c}"
                )
                self.categorical_vars[c] = var

    def add_binary_constraints(self):
        gurobi_model = self.gurobi_model
        feature_encoder = self.feature_encoder
        tree_ensemble = self.tree_ensemble
        for f in feature_encoder.binary_features:
            for t, tree in enumerate(tree_ensemble):
                for n in tree.node_split_on(f):
                    l = tree.left[n]
                    r = tree.right[n]
                    cons = gurobi_model.addConstr(
                        self.binary_vars[f] <=
                        1 - self.flow_vars[t, l],
                        name=f"binary_left_{f}_{t}_{n}"
                    )
                    self.binary_left_constraints[f, t, n] = cons
                    cons = gurobi_model.addConstr(
                        self.binary_vars[f] >=
                        self.flow_vars[t, r],
                        name=f"binary_right_{f}_{t}_{n}"
                    )
                    self.binary_right_constraints[f, t, n] = cons

    def add_discrete_constraints(self):
        gurobi_model = self.gurobi_model
        feature_encoder = self.feature_encoder
        tree_ensemble = self.tree_ensemble
        for f in feature_encoder.discrete_features:
            values = feature_encoder.values[f]
            for i, v in enumerate(values):
                if i == 0:
                    cons = gurobi_model.addConstr(
                        self.discrete_vars[f, i] == 1.0,
                        name=f"discrete_logical_{f}_{i}"
                    )
                    self.discrete_logical_constraints[f, i] = cons
                else:
                    cons = gurobi_model.addConstr(
                        self.discrete_vars[f, i] <=
                        self.discrete_vars[f, i-1],
                        name=f"discrete_logical_{f}_{i}"
                    )
                    self.discrete_logical_constraints[f, i] = cons
                for t, tree in enumerate(tree_ensemble):
                    for n in tree.node_split_on(f):
                        if v == tree.threshold[n]:
                            l = tree.left[n]
                            r = tree.right[n]    
                            cons = gurobi_model.addConstr(
                                self.discrete_vars[f, i] <=
                                1 - self.flow_vars[t, l],
                                name=f"discrete_left_{f}_{i}_{t}_{n}"
                            )
                            self.discrete_left_constraints[f, i, t, n] = cons
                            cons = gurobi_model.addConstr(
                                self.discrete_vars[f, i] >=
                                self.flow_vars[t, r],
                                name=f"discrete_right_{f}_{i}_{t}_{n}"
                            )
                            self.discrete_right_constraints[f, i, t, n] = cons

    def add_continuous_constraints(self):
        gurobi_model = self.gurobi_model
        feature_encoder = self.feature_encoder
        tree_ensemble = self.tree_ensemble
        for f in feature_encoder.continuous_features:
            levels = tree_ensemble.numerical_levels[f]
            for i, v in enumerate(levels[:-1]):
                if i == 0:
                    cons = gurobi_model.addConstr(
                        self.continuous_vars[f, i] == 1.0,
                        name=f"continuous_logical_{f}_{i}"
                    )
                    self.continuous_logical_constraints[f, i] = cons
                else:
                    cons = gurobi_model.addConstr(
                        self.continuous_vars[f, i] <=
                        self.continuous_vars[f, i-1],
                        name=f"continuous_logical_{f}_{i}"
                    )
                    self.continuous_logical_constraints[f, i] = cons
                    for t, tree in enumerate(tree_ensemble):
                        for n in tree.node_split_on(f):
                            if v == tree.threshold[n]:
                                l = tree.left[n]
                                r = tree.right[n]
                                cons = gurobi_model.addConstr(
                                    self.continuous_vars[f, i] <=
                                    1 - self.flow_vars[t, l],
                                    name=f"continuous_left_{f}_{i}_{t}_{n}"
                                )
                                self.continuous_left_constraints[f, i, t, n] = cons
                                cons = gurobi_model.addConstr(
                                    self.continuous_vars[f, i] >=
                                    self.flow_vars[t, r],
                                    name=f"continuous_right_{f}_{i}_{t}_{n}"
                                )
                                self.continuous_right_constraints[f, i, t, n] = cons

    def add_categorical_constraints(self):
        gurobi_model = self.gurobi_model
        feature_encoder = self.feature_encoder
        tree_ensemble = self.tree_ensemble
        for f in feature_encoder.categorical_features:
            cons = gurobi_model.addConstr(
                gp.quicksum(
                    self.categorical_vars[c]
                    for c in feature_encoder.categories[f]
                ) == 1,
                name=f"categorical_logical_{f}"
            )
            self.categorical_logical_constraints[f] = cons
            for c in feature_encoder.categories[f]:
                for t, tree in enumerate(tree_ensemble):
                    for n in tree.node_split_on(f):
                        if c == tree.category[n]:
                            l = tree.left[n]
                            r = tree.right[n]
                            cons = gurobi_model.addConstr(
                                self.categorical_vars[c] <=
                                1 - self.flow_vars[t, l],
                                name=f"categorical_left_{c}_{t}_{n}"
                            )
                            self.categorical_left_constraints[c, t, n] = cons
                            cons = gurobi_model.addConstr(
                                self.categorical_vars[c] >=
                                self.flow_vars[t, r],
                                name=f"categorical_right_{c}_{t}_{n}"
                            )
                            self.categorical_right_constraints[c, t, n] = cons

    def add_prob_vars(self):
        gurobi_model = self.gurobi_model
        tree_ensemble = self.tree_ensemble
        k = tree_ensemble.n_classes
        for c in range(k):
            var = gurobi_model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"prob_{c}"
            )
            self.prob_vars[c] = var

    def add_prob_constraints(self):
        gurobi_model = self.gurobi_model
        tree_ensemble = self.tree_ensemble
        k = tree_ensemble.n_classes
        for c in range(k):
            cons = gurobi_model.addConstr(
                self.prob_vars[c] ==
                gp.quicksum(
                    self.weights[t]
                    *self.flow_vars[t, n]
                    *tree.prob[n][c]
                    for t, tree in enumerate(tree_ensemble)
                    for n in tree.leaves
                ),
                name=f"prob_{c}"
            )
            self.prob_constraints[c] = cons

    def build(self):
        gurobi_model = gp.Model("FIPEOracle")
        self.gurobi_model = gurobi_model
        self.flow_vars = gp.tupledict()
        self.branch_vars = gp.tupledict()
        self.root_constraints = gp.tupledict()
        self.flow_constraints = gp.tupledict()
        self.branch_to_left_constraints = gp.tupledict()
        self.branch_to_right_constraints = gp.tupledict()
        
        self.binary_vars = gp.tupledict()
        self.discrete_vars = gp.tupledict()
        self.continuous_vars = gp.tupledict()
        self.categorical_vars = gp.tupledict()
        
        self.binary_left_constraints = gp.tupledict()
        self.binary_right_constraints = gp.tupledict()
        self.discrete_left_constraints = gp.tupledict()
        self.discrete_right_constraints = gp.tupledict()
        self.discrete_logical_constraints = gp.tupledict()
        self.continuous_left_constraints = gp.tupledict()
        self.continuous_right_constraints = gp.tupledict()
        self.continuous_logical_constraints = gp.tupledict()
        self.categorical_left_constraints = gp.tupledict()
        self.categorical_right_constraints = gp.tupledict()
        self.categorical_logical_constraints = gp.tupledict()
        
        self.prob_vars = gp.tupledict()
        self.prob_constraints = gp.tupledict()
        self.prune_prob_vars = gp.tupledict()
        self.prune_prob_constraints = gp.tupledict()
        self.majority_class_constraints = gp.tupledict()

        self.build_trees()
        self.build_features()
        
        self.add_prob_vars()
        self.add_prob_constraints()

    def add_prune_prob_vars(self, classes):
        gurobi_model = self.gurobi_model
        for c in classes:
            var = gurobi_model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                name=f"prune_prob_{c}"
            )
            self.prune_prob_vars[c] = var

    def add_prune_prob_constraints(self, active, classes: int | list[int]):
        gurobi_model = self.gurobi_model
        active = np.array(active)
        if isinstance(classes, int):
            classes = [classes]
        for c in classes:
            cons = gurobi_model.addConstr(
                self.prune_prob_vars[c] ==
                gp.quicksum(
                    self.weights[t]
                    *active[t]
                    *self.flow_vars[t, n]
                    *tree.prob[n][c]
                    for t, tree in enumerate(self.tree_ensemble)
                    for n in tree.leaves
                ),
                name=f"prune_prob_{c}"
            )
            self.prune_prob_constraints[c] = cons

    def add_majority_class_constraints(self, mc: int):
        gurobi_model = self.gurobi_model
        k = self.tree_ensemble.n_classes
        wm = self.weights.min()
        eps = self.eps
        for c in range(k):
            if c == mc:
                continue
            rhs = (0.0 if c > mc else eps*wm)
            cons = gurobi_model.addConstr(
                self.prob_vars[mc] - self.prob_vars[c]
                >= rhs,
                name=f"majority_class_{c}"
            )
            self.majority_class_constraints[c] = cons

    def add_objective(self, mc: int, c: int):
        gurobi_model = self.gurobi_model
        gurobi_model.setObjective(
            self.prune_prob_vars[c] - self.prune_prob_vars[mc],
            GRB.MAXIMIZE
        )

    def add_active(self, active, mc: int, c: int):
        self.add_prune_prob_vars([mc, c])
        self.add_prune_prob_constraints(active, [mc, c])
        self.add_majority_class_constraints(mc)
        self.add_objective(mc, c)

    def reset(self):
        self.gurobi_model.remove(self.prune_prob_vars)
        self.gurobi_model.remove(self.prune_prob_constraints)
        self.gurobi_model.remove(self.majority_class_constraints)

    def optimize(self):
        self.gurobi_model.optimize()

    def set_gurobi_parameter(self, param, value):
        self.gurobi_model.setParam(param, value)

    def get_solutions(self, cutoff=1.0):
        if self.gurobi_model.SolCount == 0:
            logger.warning("When solving the FIPE Oracle problem, no solution was found.")
            return []

        solutions = []
        feature_encoder = self.feature_encoder
        for i in range(self.gurobi_model.SolCount):
            self.gurobi_model.setParam(GRB.Param.SolutionNumber, i)
            obj = self.gurobi_model.PoolObjVal
            if obj < cutoff:
                continue

            solution = dict()
            for f in feature_encoder.binary_features:
                solution[f] = self.binary_vars[f].Xn > 0.5
            for f in feature_encoder.discrete_features:
                values = feature_encoder.values[f]
                j = 0
                while j < len(values) and self.discrete_vars[f, j].Xn > 0.5:
                    j += 1
            
                assert j < len(values)
                solution[f] = values[j-1]
        
            for f in feature_encoder.continuous_features:
                levels = self.tree_ensemble.numerical_levels[f]
                j = 0
                while j < len(levels)-1 and self.continuous_vars[f, j].Xn > 0.5:
                    j += 1
                if j == len(levels)-1:
                    solution[f] = levels[j]
                else:
                    solution[f] = (levels[j-1] + levels[j]) / 2.0

            for f in feature_encoder.categorical_features:
                categories = feature_encoder.categories[f]
                for c in categories:
                    solution[c] = self.categorical_vars[c].Xn > 0.5
            solutions.append(solution)
        cols = feature_encoder.columns
        return pd.DataFrame(solutions, columns=cols).values

class FIPEPrunerFull:
    pruner: FIPEPruner
    oracle: FIPEOracle
    max_iter: int

    def __init__(self, E, w, feature_encoder, **kwargs):
        tree_ensemble = TreeEnsemble(E, feature_encoder, **kwargs)
        self.pruner = FIPEPruner(E, w, **kwargs)
        self.oracle = FIPEOracle(feature_encoder, tree_ensemble, w, **kwargs)
        self.max_iter = kwargs.get("max_iter", 100)

    def build(self):
        self.pruner.build()
        self.oracle.build()

    def prune(self, X):
        self.pruner.add_constraints(X)
        self.pruner.prune()
        
        # Call the FIPE Oracle
        active = deepcopy(self.pruner.active)
        n_classes = self.oracle.tree_ensemble.n_classes
        
        it = 0
        while True:
            for c1 in range(n_classes):
                for c2 in range(n_classes):
                    if c1 == c2:
                        continue

                    self.oracle.add_active(active, c1, c2)
                    self.oracle.optimize()
                    X = self.oracle.get_solutions()
                    if len(X) == 0:
                        continue
                    self.pruner.add_constraints(X)
                    self.oracle.reset()
            
            self.pruner.prune()
            if np.isclose(active, self.pruner.active).all():
                break

            active = self.pruner.active
            it += 1
            if it >= self.max_iter:
                break 
