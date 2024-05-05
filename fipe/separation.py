from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from sklearn.ensemble._iforest import _average_path_length  # type: ignore

from .tree import TreeEnsemble
from .encoding import FeatureEncoder
from ._predict import predict_proba
from .typing import numeric

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def add_tree_ensemble_flow_vars(
    model: gp.Model,
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    name: str = "flow"
):
    for t, tree in enumerate(tree_ensemble):
        for n in tree:
            var = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{name}_{t}_{n}"
            )
            flow_vars[t, n] = var


def add_tree_ensemble_branch_vars(
    model: gp.Model,
    tree_ensemble: TreeEnsemble,
    branch_vars: gp.tupledict[tuple[int, int], gp.Var],
    name: str = "branch"
):
    for t, tree in enumerate(tree_ensemble):
        for d in range(tree.max_depth):
            var = model.addVar(
                vtype=GRB.BINARY,
                name=f"{name}_{t}_{d}"
            )
            branch_vars[t, d] = var


def add_tree_ensemble_root_constraints(
    model: gp.Model,
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    root_constraints: gp.tupledict[int, gp.Constr],
    name: str = "root"
):
    for t, tree in enumerate(tree_ensemble):
        root = tree.root
        cons = model.addConstr(
            flow_vars[t, root] == 1.0,
            name=f"{name}_{t}"
        )
        root_constraints[t] = cons


def add_tree_ensemble_flow_constraints(
    model: gp.Model,
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    flow_constraints: gp.tupledict[tuple[int, int], gp.Constr],
    name: str = "flow"
):
    for t, tree in enumerate(tree_ensemble):
        for n in tree.internal_nodes:
            left = tree.left[n]
            right = tree.right[n]
            cons = model.addConstr(
                flow_vars[t, left] + flow_vars[t, right]
                == flow_vars[t, n],
                name=f"{name}_{t}_{n}"
            )
            flow_constraints[t, n] = cons


def add_tree_ensemble_branch_constraints(
    model: gp.Model,
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    branch_vars: gp.tupledict[tuple[int, int], gp.Var],
    branch_to_left_constraints: gp.tupledict[tuple[int, int], gp.Constr],
    branch_to_right_constraints: gp.tupledict[tuple[int, int], gp.Constr],
    name: str = "branch"
):
    for t, tree in enumerate(tree_ensemble):
        for n in tree.internal_nodes:
            d = tree.node_depth[n]
            left = tree.left[n]
            right = tree.right[n]
            cons = model.addConstr(
                flow_vars[t, left] <= branch_vars[t, d],
                name=f"{name}_to_left_{t}_{n}"
            )
            branch_to_left_constraints[t, n] = cons
            cons = model.addConstr(
                flow_vars[t, right] <= 1 - branch_vars[t, d],
                name=f"{name}_to_right_{t}_{n}"
            )
            branch_to_right_constraints[t, n] = cons


def add_tree_ensemble_vars_and_constraints(
    model: gp.Model,
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    branch_vars: gp.tupledict[tuple[int, int], gp.Var],
    root_constraints: gp.tupledict[int, gp.Constr],
    flow_constraints: gp.tupledict[tuple[int, int], gp.Constr],
    branch_to_left_constraints: gp.tupledict[tuple[int, int], gp.Constr],
    branch_to_right_constraints: gp.tupledict[tuple[int, int], gp.Constr],
    prefix: str = "",
    prefix_sep: str = "_"
):
    add_tree_ensemble_flow_vars(
        model, tree_ensemble, flow_vars, name=f"{prefix}{prefix_sep}flow"
    )
    add_tree_ensemble_branch_vars(
        model, tree_ensemble, branch_vars, name=f"{prefix}{prefix_sep}branch"
    )
    add_tree_ensemble_root_constraints(
        model, tree_ensemble, flow_vars, root_constraints,
        name=f"{prefix}{prefix_sep}root"
    )
    add_tree_ensemble_flow_constraints(
        model, tree_ensemble, flow_vars, flow_constraints,
        name=f"{prefix}{prefix_sep}flow"
    )
    add_tree_ensemble_branch_constraints(
        model, tree_ensemble, flow_vars, branch_vars,
        branch_to_left_constraints, branch_to_right_constraints,
        name=f"{prefix}{prefix_sep}branch"
    )


def add_feature_binary_vars(
    model: gp.Model,
    binary_features: list[str],
    binary_vars: gp.tupledict[str, gp.Var],
    name: str = "binary"
):
    for f in binary_features:
        var = model.addVar(
            vtype=GRB.BINARY,
            name=f"{name}_{f}"
        )
        binary_vars[f] = var


def add_feature_discrete_vars(
    model: gp.Model,
    discrete_features: list[str],
    discrete_values: dict[str, list[numeric]],
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var],
    name: str = "discrete"
):
    for f in discrete_features:
        n = len(discrete_values[f])
        for v in range(n):
            var = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{name}_{f}_{v}"
            )
            discrete_vars[f, v] = var


def add_feature_continuous_vars(
    model: gp.Model,
    continuous_features: list[str],
    continuous_levels: dict[str, list[numeric]],
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var],
    name: str = "continuous"
):
    for f in continuous_features:
        n = len(continuous_levels[f]) - 1
        for i in range(n):
            var = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{name}_{f}_{i}"
            )
            continuous_vars[f, i] = var


def add_feature_categorical_vars(
    model: gp.Model,
    categorical_features: list[str],
    categorical_categories: dict[str, list[str]],
    categorical_vars: gp.tupledict[str, gp.Var],
    name: str = "categorical"
):
    for f in categorical_features:
        for c in categorical_categories[f]:
            var = model.addVar(
                vtype=GRB.BINARY,
                name=f"{name}_{c}"
            )
            categorical_vars[c] = var


def add_feature_binary_left_constraints(
    model: gp.Model,
    binary_features: list[str],
    binary_vars: gp.tupledict[str, gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    binary_left_constraints: gp.tupledict[tuple[str, int, int], gp.Constr],
    name: str = "binary_left"
):
    for f in binary_features:
        for t, tree in enumerate(tree_ensemble):
            for n in tree.node_split_on(f):
                left = tree.left[n]
                cons = model.addConstr(
                    binary_vars[f] <= 1 - flow_vars[t, left],
                    name=f"{name}_{f}_{t}_{n}"
                )
                binary_left_constraints[f, t, n] = cons


def add_feature_binary_right_constraints(
    model: gp.Model,
    binary_features: list[str],
    binary_vars: gp.tupledict[str, gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    binary_right_constraints: gp.tupledict[tuple[str, int, int], gp.Constr],
    name: str = "binary_right"
):
    for f in binary_features:
        for t, tree in enumerate(tree_ensemble):
            for n in tree.node_split_on(f):
                right = tree.right[n]
                cons = model.addConstr(
                    binary_vars[f] >= flow_vars[t, right],
                    name=f"{name}_{f}_{t}_{n}"
                )
                binary_right_constraints[f, t, n] = cons


def add_feature_binary_constraints(
    model: gp.Model,
    binary_features: list[str],
    binary_vars: gp.tupledict[str, gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    binary_left_constraints: gp.tupledict[tuple[str, int, int], gp.Constr],
    binary_right_constraints: gp.tupledict[tuple[str, int, int], gp.Constr],
    prefix: str = "binary",
    prefix_sep: str = "_"
):
    add_feature_binary_left_constraints(
        model, binary_features, binary_vars, tree_ensemble,
        flow_vars, binary_left_constraints,
        name=f"{prefix}{prefix_sep}left"
    )
    add_feature_binary_right_constraints(
        model, binary_features, binary_vars, tree_ensemble,
        flow_vars, binary_right_constraints,
        name=f"{prefix}{prefix_sep}right"
    )


def add_feature_discrete_left_constraints(
    model: gp.Model,
    discrete_features: list[str],
    discrete_values: dict[str, list[numeric]],
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    discrete_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    name: str = "discrete_left"
):
    for f in discrete_features:
        values = discrete_values[f]
        for i, v in enumerate(values):
            for t, tree in enumerate(tree_ensemble):
                for n in tree.node_split_on(f):
                    if v == tree.threshold[n]:
                        left = tree.left[n]
                        cons = model.addConstr(
                            discrete_vars[f, i]
                            <= 1 - flow_vars[t, left],
                            name=f"{name}_{f}_{i}_{t}_{n}"
                        )
                        discrete_left_constraints[f, i, t, n] = cons


def add_feature_discrete_right_constraints(
    model: gp.Model,
    discrete_features: list[str],
    discrete_values: dict[str, list[numeric]],
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    discrete_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    name: str = "discrete_right"
):
    for f in discrete_features:
        values = discrete_values[f]
        for i, v in enumerate(values):
            for t, tree in enumerate(tree_ensemble):
                for n in tree.node_split_on(f):
                    if v == tree.threshold[n]:
                        right = tree.right[n]
                        cons = model.addConstr(
                            discrete_vars[f, i]
                            >= flow_vars[t, right],
                            name=f"{name}_{f}_{i}_{t}_{n}"
                        )
                        discrete_right_constraints[f, i, t, n] = cons


def add_feature_discrete_logical_constraints(
    model: gp.Model,
    discrete_features: list[str],
    discrete_values: dict[str, list[numeric]],
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var],
    discrete_logical_constraints: gp.tupledict[tuple[str, int], gp.Constr],
    name: str = "discrete_logical"
):
    for f in discrete_features:
        n = len(discrete_values[f])
        for i in range(n):
            if i == 0:
                cons = model.addConstr(
                    discrete_vars[f, i] == 1.0,
                    name=f"{name}_{f}_{i}"
                )
                discrete_logical_constraints[f, i] = cons
            else:
                cons = model.addConstr(
                    discrete_vars[f, i]
                    <= discrete_vars[f, i-1],
                    name=f"{name}_{f}_{i}"
                )
                discrete_logical_constraints[f, i] = cons


def add_feature_discrete_constraints(
    model: gp.Model,
    discrete_features: list[str],
    discrete_values: dict[str, list[numeric]],
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    discrete_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    discrete_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    discrete_logical_constraints: gp.tupledict[
        tuple[str, int], gp.Constr],
    prefix: str = "discrete",
    prefix_sep: str = "_"
):
    add_feature_discrete_left_constraints(
        model, discrete_features, discrete_values, discrete_vars,
        tree_ensemble, flow_vars, discrete_left_constraints,
        name=f"{prefix}{prefix_sep}left"
    )
    add_feature_discrete_right_constraints(
        model, discrete_features, discrete_values, discrete_vars,
        tree_ensemble, flow_vars, discrete_right_constraints,
        name=f"{prefix}{prefix_sep}right"
    )
    add_feature_discrete_logical_constraints(
        model, discrete_features, discrete_values, discrete_vars,
        discrete_logical_constraints,
        name=f"{prefix}{prefix_sep}logical"
    )


def add_feature_continuous_left_constraints(
    model: gp.Model,
    continuous_features: list[str],
    continuous_levels: dict[str, list[numeric]],
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    continuous_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    name: str = "continuous_left"
):
    for f in continuous_features:
        levels = continuous_levels[f]
        for i, v in enumerate(levels[:-1]):
            for t, tree in enumerate(tree_ensemble):
                for n in tree.node_split_on(f):
                    if v == tree.threshold[n]:
                        left = tree.left[n]
                        cons = model.addConstr(
                            continuous_vars[f, i]
                            <= 1 - flow_vars[t, left],
                            name=f"{name}_{f}_{i}_{t}_{n}"
                        )
                        continuous_left_constraints[f, i, t, n] = cons


def add_feature_continuous_right_constraints(
    model: gp.Model,
    continuous_features: list[str],
    continuous_levels: dict[str, list[numeric]],
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    continuous_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    name: str = "continuous_right"
):
    for f in continuous_features:
        levels = continuous_levels[f]
        for i, v in enumerate(levels[:-1]):
            for t, tree in enumerate(tree_ensemble):
                for n in tree.node_split_on(f):
                    if v == tree.threshold[n]:
                        right = tree.right[n]
                        cons = model.addConstr(
                            continuous_vars[f, i]
                            >= flow_vars[t, right],
                            name=f"{name}_{f}_{i}_{t}_{n}"
                        )
                        continuous_right_constraints[f, i, t, n] = cons


def add_feature_continuous_logical_constraints(
    model: gp.Model,
    continuous_features: list[str],
    continuous_levels: dict[str, list[numeric]],
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var],
    continuous_logical_constraints: gp.tupledict[tuple[str, int], gp.Constr],
    name: str = "continuous_logical"
):
    for f in continuous_features:
        levels = continuous_levels[f]
        n = len(levels) - 1
        for i in range(n):
            if i == 0:
                cons = model.addConstr(
                    continuous_vars[f, i] == 1.0,
                    name=f"{name}_{f}_{i}"
                )
                continuous_logical_constraints[f, i] = cons
            else:
                cons = model.addConstr(
                    continuous_vars[f, i]
                    <= continuous_vars[f, i-1],
                    name=f"{name}_{f}_{i}"
                )
                continuous_logical_constraints[f, i] = cons


def add_feature_continuous_constraints(
    model: gp.Model,
    continuous_features: list[str],
    continuous_levels: dict[str, list[numeric]],
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    continuous_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    continuous_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr],
    continuous_logical_constraints: gp.tupledict[
        tuple[str, int], gp.Constr],
    prefix: str = "continuous",
    prefix_sep: str = "_"
):
    add_feature_continuous_left_constraints(
        model, continuous_features, continuous_levels, continuous_vars,
        tree_ensemble, flow_vars, continuous_left_constraints,
        name=f"{prefix}{prefix_sep}left"
    )
    add_feature_continuous_right_constraints(
        model, continuous_features, continuous_levels, continuous_vars,
        tree_ensemble, flow_vars, continuous_right_constraints,
        name=f"{prefix}{prefix_sep}right"
    )
    add_feature_continuous_logical_constraints(
        model, continuous_features, continuous_levels, continuous_vars,
        continuous_logical_constraints,
        name=f"{prefix}{prefix_sep}logical"
    )


def add_feature_categorical_left_constraints(
    model: gp.Model,
    categorical_features: list[str],
    categorical_categories: dict[str, list[str]],
    categorical_vars: gp.tupledict[str, gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    categorical_left_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr],
    name: str = "categorical_left"
):
    for f in categorical_features:
        for c in categorical_categories[f]:
            for t, tree in enumerate(tree_ensemble):
                for n in tree.node_split_on(f):
                    left = tree.left[n]
                    cons = model.addConstr(
                        categorical_vars[c]
                        <= 1 - flow_vars[t, left],
                        name=f"{name}_{f}_{c}_{t}_{n}"
                    )
                    categorical_left_constraints[c, t, n] = cons


def add_feature_categorical_right_constraints(
    model: gp.Model,
    categorical_features: list[str],
    categorical_categories: dict[str, list[str]],
    categorical_vars: gp.tupledict[str, gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    categorical_right_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr],
    name: str = "categorical_right"
):
    for f in categorical_features:
        for c in categorical_categories[f]:
            for t, tree in enumerate(tree_ensemble):
                for n in tree.node_split_on(f):
                    right = tree.right[n]
                    cons = model.addConstr(
                        categorical_vars[c]
                        >= flow_vars[t, right],
                        name=f"{name}_{f}_{c}_{t}_{n}"
                    )
                    categorical_right_constraints[c, t, n] = cons


def add_feature_categorical_logical_constraints(
    model: gp.Model,
    categorical_features: list[str],
    categorical_categories: dict[str, list[str]],
    categorical_vars: gp.tupledict[str, gp.Var],
    categorical_logical_constraints: gp.tupledict[str, gp.Constr],
    name: str = "categorical_logical"
):
    for f in categorical_features:
        for c in categorical_categories[f]:
            cons = model.addConstr(
                categorical_vars[c] == 1.0,
                name=f"{name}_{c}"
            )
            categorical_logical_constraints[c] = cons


def add_feature_categorical_constraints(
    model: gp.Model,
    categorical_features: list[str],
    categorical_categories: dict[str, list[str]],
    categorical_vars: gp.tupledict[str, gp.Var],
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    categorical_left_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr],
    categorical_right_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr],
    categorical_logical_constraints: gp.tupledict[str, gp.Constr],
    prefix: str = "categorical",
    prefix_sep: str = "_"
):
    add_feature_categorical_left_constraints(
        model, categorical_features, categorical_categories, categorical_vars,
        tree_ensemble, flow_vars, categorical_left_constraints,
        name=f"{prefix}{prefix_sep}left"
    )
    add_feature_categorical_right_constraints(
        model, categorical_features, categorical_categories, categorical_vars,
        tree_ensemble, flow_vars, categorical_right_constraints,
        name=f"{prefix}{prefix_sep}right"
    )
    add_feature_categorical_logical_constraints(
        model, categorical_features, categorical_categories, categorical_vars,
        categorical_logical_constraints,
        name=f"{prefix}{prefix_sep}logical"
    )


def add_prob_vars(
    model: gp.Model,
    prob_vars: gp.tupledict[int, gp.Var],
    classes: int | list[int],
    name: str = "prob"
):
    if isinstance(classes, int):
        classes = [classes]
    for c in classes:
        var = model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name=f"{name}_{c}"
        )
        prob_vars[c] = var


def build_weighted_prob_expr(
    tree_ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    weights: np.ndarray,
    c: int
):
    return gp.quicksum(
        weights[t]
        * tree.prob[n][c]
        * flow_vars[t, n]
        for t, tree in enumerate(tree_ensemble)
        for n in tree.leaves
    )


def add_prob_constraints(
    model: gp.Model,
    tree_ensemble: TreeEnsemble,
    classes: int | list[int],
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    prob_vars: gp.tupledict[int, gp.Var],
    weights: np.ndarray,
    prob_constraints: gp.tupledict[int, gp.Constr],
    prefix: str = "prob",
    prefix_sep: str = "_"
):
    if isinstance(classes, int):
        classes = [classes]

    for c in classes:
        expr = build_weighted_prob_expr(
            tree_ensemble, flow_vars, weights, c
        )
        cons = model.addConstr(
            prob_vars[c] == expr,
            name=f"{prefix}{prefix_sep}{c}"
        )
        prob_constraints[c] = cons


def add_prob_objective(
    model: gp.Model,
    prob_vars: gp.tupledict[int, gp.Var],
    c1: int,
    c2: int
):
    model.setObjective(
        prob_vars[c1] - prob_vars[c2],
        GRB.MAXIMIZE
    )


def add_majority_class(
    model: gp.Model,
    prob_vars: gp.tupledict[int, gp.Var],
    majority_class: int,
    wm: float,
    majority_class_constraints: gp.tupledict[int, gp.Constr],
    name: str = "majority"
):
    for c in prob_vars.keys():
        if c == majority_class:
            continue
        rhs = (0.0 if majority_class < c else wm)
        cons = model.addConstr(
            prob_vars[majority_class] - prob_vars[c] >= rhs,
            name=f"{name}_{c}"
        )
        majority_class_constraints[c] = cons


def add_isolation_plausibility_constraints(
    model: gp.Model,
    ensemble: TreeEnsemble,
    flow_vars: gp.tupledict[tuple[int, int], gp.Var],
    max_samples: int,
    anomaly_var: gp.Var,
    offset: float,
    name: str = "isolation_plausibility"
) -> tuple[gp.Constr, gp.Constr]:

    m = len(ensemble)
    d = m * _average_path_length([max_samples])[0]

    lhs = gp.quicksum(
        (tree.node_depth[n]
         + _average_path_length(
             [tree.n_samples[n]])[0]
         ) * flow_vars[t, n]
        for t, tree in enumerate(ensemble)
        for n in tree.leaves
    ) / d

    min_score = -np.log2(-offset)
    return model.addConstr(
        lhs == anomaly_var,
        name=name
    ), model.addConstr(
        anomaly_var >= min_score,
        name=f"{name}_min_score"
    )


def get_binary_feature_values(
    binary_features: list[str],
    binary_vars: gp.tupledict[str, gp.Var],
    best: bool = False
):
    x = dict()
    for f in binary_features:
        if best:
            x[f] = binary_vars[f].X
        else:
            x[f] = binary_vars[f].Xn
    return x


def get_discrete_feature_values(
    discrete_features: list[str],
    discrete_values: dict[str, list[numeric]],
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var],
    best: bool = False
):
    x = dict()
    for f in discrete_features:
        n = len(discrete_values[f])
        j = 0
        while j < n:
            if best:
                if discrete_vars[f, j].X < 0.5:
                    break
            else:
                if discrete_vars[f, j].Xn < 0.5:
                    break
            j += 1
        if j == n:
            j -= 1
        x[f] = discrete_values[f][j]
    return x


def get_continuous_feature_values(
    continuous_features: list[str],
    continuous_levels: dict[str, list[numeric]],
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var],
    best: bool = False
):
    x = dict()
    for f in continuous_features:
        levels = continuous_levels[f]
        n = len(levels) - 1
        j = 0
        while j < n:
            if best:
                if continuous_vars[f, j].X < 0.5:
                    break
            else:
                if continuous_vars[f, j].Xn < 0.5:
                    break
            j += 1
        if j == n:
            j -= 1
        x[f] = (levels[j-1] + levels[j]) / 2.0
    return x


def get_categorical_feature_values(
    categorical_features: list[str],
    categorical_categories: dict[str, list[str]],
    categorical_vars: gp.tupledict[str, gp.Var],
    best: bool = False
):
    x = dict()
    for f in categorical_features:
        for c in categorical_categories[f]:
            if best:
                x[c] = categorical_vars[c].X
            else:
                x[c] = categorical_vars[c].Xn
    return x


def get_feature_values(
    binary_features: list[str],
    binary_vars: gp.tupledict[str, gp.Var],
    discrete_features: list[str],
    discrete_values: dict[str, list[numeric]],
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var],
    continuous_features: list[str],
    continuous_levels: dict[str, list[numeric]],
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var],
    categorical_features: list[str],
    categorical_categories: dict[str, list[str]],
    categorical_vars: gp.tupledict[str, gp.Var],
    best: bool = False
):
    x = dict()
    x.update(get_binary_feature_values(
        binary_features, binary_vars, best))
    x.update(get_discrete_feature_values(
        discrete_features, discrete_values, discrete_vars, best))
    x.update(get_continuous_feature_values(
        continuous_features, continuous_levels, continuous_vars, best))
    x.update(get_categorical_feature_values(
        categorical_features, categorical_categories, categorical_vars, best))
    return x


def to_array_values(
    columns: list[str],
    values: dict[str, numeric] | list[dict[str, numeric]]
):
    if not isinstance(values, list):
        values = [values]

    X = pd.DataFrame(values, columns=columns)
    return X


class BaseOracle(ABC):
    @abstractmethod
    def separate(self, active):
        raise NotImplementedError("The method `separate` must be implemented.")


class FIPEOracle:
    feature_encoder: FeatureEncoder
    tree_ensemble: TreeEnsemble
    isolation_ensemble: TreeEnsemble

    gurobi_model: gp.Model

    # Trees:
    flow_vars: gp.tupledict[tuple[int, int], gp.Var]
    branch_vars: gp.tupledict[tuple[int, int], gp.Var]
    root_constraints: gp.tupledict[int, gp.Constr]
    flow_constraints: gp.tupledict[tuple[int, int], gp.Constr]
    branch_to_left_constraints: gp.tupledict[tuple[int, int], gp.Constr]
    branch_to_right_constraints: gp.tupledict[tuple[int, int], gp.Constr]

    # Isolation:
    isolation_flow_vars: gp.tupledict[tuple[int, int], gp.Var]
    isolation_branch_vars: gp.tupledict[tuple[int, int], gp.Var]
    isolation_root_constraints: gp.tupledict[int, gp.Constr]
    isolation_flow_constraints: gp.tupledict[
        tuple[int, int], gp.Constr]
    isolation_branch_to_left_constraints: gp.tupledict[
        tuple[int, int], gp.Constr]
    isolation_branch_to_right_constraints: gp.tupledict[
        tuple[int, int], gp.Constr]

    # Features:
    # Variables:
    binary_vars: gp.tupledict[str, gp.Var]
    discrete_vars: gp.tupledict[tuple[str, int], gp.Var]
    continuous_vars: gp.tupledict[tuple[str, int], gp.Var]
    categorical_vars: gp.tupledict[str, gp.Var]

    # Constraints:
    binary_left_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]
    binary_right_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]
    isolation_binary_left_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]
    isolation_binary_right_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]

    discrete_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    discrete_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    discrete_logical_constraints: gp.tupledict[
        tuple[str, int], gp.Constr]
    isolation_discrete_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    isolation_discrete_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    isolation_discrete_logical_constraints: gp.tupledict[
        tuple[str, int], gp.Constr]

    continuous_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    continuous_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    continuous_logical_constraints: gp.tupledict[
        tuple[str, int], gp.Constr]
    isolation_continuous_left_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    isolation_continuous_right_constraints: gp.tupledict[
        tuple[str, int, int, int], gp.Constr]
    isolation_continuous_logical_constraints: gp.tupledict[
        tuple[str, int], gp.Constr]

    categorical_left_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]
    categorical_right_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]
    categorical_logical_constraints: gp.tupledict[str, gp.Constr]
    isolation_categorical_left_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]
    isolation_categorical_right_constraints: gp.tupledict[
        tuple[str, int, int], gp.Constr]
    isolation_categorical_logical_constraints: gp.tupledict[str, gp.Constr]

    # Probabilities:
    prob_vars: gp.tupledict[int, gp.Var]
    prob_constraints: gp.tupledict[int, gp.Constr]
    anomaly_var: gp.Var
    anomaly_constraint: gp.Constr
    isolation_constraint = gp.Constr

    active_prob_vars: gp.tupledict[int, gp.Var]
    active_prob_constraints: gp.tupledict[int, gp.Constr]
    majority_class_constraints: gp.tupledict[int, gp.Constr]

    continuous_levels: dict[str, list[numeric]]
    eps: float
    max_samples: int
    offset: float
    c1: int
    c2: int

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        tree_ensemble: TreeEnsemble,
        isolation_ensemble: TreeEnsemble,
        weights,
        max_samples: int,
        offset: float,
        **kwargs
    ):
        self.feature_encoder = feature_encoder
        self.tree_ensemble = tree_ensemble
        self.isolation_ensemble = isolation_ensemble
        self.weights = np.array(weights)
        self.max_samples = max_samples
        self.offset = offset
        self.continuous_levels = dict()
        self.merge_continuous_levels()
        self.eps = kwargs.get("eps", 1.0)

    def merge_continuous_levels(self):
        for f in self.feature_encoder.continuous_features:
            levels = set(self.tree_ensemble.numerical_levels[f])
            levels.update(self.isolation_ensemble.numerical_levels[f])
            levels = sorted(levels)
            self.continuous_levels[f] = list(levels)

    def add_binary_vars(self):
        add_feature_binary_vars(
            self.gurobi_model,
            self.feature_encoder.binary_features,
            self.binary_vars, name="binary"
        )

    def add_discrete_vars(self):
        add_feature_discrete_vars(
            self.gurobi_model,
            self.feature_encoder.discrete_features,
            self.feature_encoder.values, self.discrete_vars,
            name="discrete"
        )

    def add_continuous_vars(self):
        add_feature_continuous_vars(
            self.gurobi_model,
            self.feature_encoder.continuous_features,
            self.continuous_levels, self.continuous_vars,
            name="continuous"
        )

    def add_categorical_vars(self):
        add_feature_categorical_vars(
            self.gurobi_model,
            self.feature_encoder.categorical_features,
            self.feature_encoder.categories, self.categorical_vars,
            name="categorical"
        )

    def add_feature_vars(self):
        self.add_binary_vars()
        self.add_discrete_vars()
        self.add_continuous_vars()
        self.add_categorical_vars()

    def add_binary_constraints(self):
        add_feature_binary_constraints(
            self.gurobi_model,
            self.feature_encoder.binary_features,
            self.binary_vars, self.tree_ensemble,
            self.flow_vars, self.binary_left_constraints,
            self.binary_right_constraints,
            prefix="tree_binary"
        )
        add_feature_binary_constraints(
            self.gurobi_model,
            self.feature_encoder.binary_features,
            self.binary_vars, self.isolation_ensemble,
            self.isolation_flow_vars, self.isolation_binary_left_constraints,
            self.isolation_binary_right_constraints,
            prefix="isolation_binary"
        )

    def add_discrete_constraints(self):
        add_feature_discrete_constraints(
            self.gurobi_model,
            self.feature_encoder.discrete_features,
            self.feature_encoder.values, self.discrete_vars,
            self.tree_ensemble, self.flow_vars,
            self.discrete_left_constraints,
            self.discrete_right_constraints,
            self.discrete_logical_constraints,
            prefix="tree_discrete"
        )
        add_feature_discrete_constraints(
            self.gurobi_model,
            self.feature_encoder.discrete_features,
            self.feature_encoder.values, self.discrete_vars,
            self.isolation_ensemble, self.isolation_flow_vars,
            self.isolation_discrete_left_constraints,
            self.isolation_discrete_right_constraints,
            self.isolation_discrete_logical_constraints,
            prefix="isolation_discrete"
        )

    def add_continuous_constraints(self):
        add_feature_continuous_constraints(
            self.gurobi_model,
            self.feature_encoder.continuous_features,
            self.continuous_levels, self.continuous_vars,
            self.tree_ensemble, self.flow_vars,
            self.continuous_left_constraints,
            self.continuous_right_constraints,
            self.continuous_logical_constraints,
            prefix="tree_continuous"
        )
        add_feature_continuous_constraints(
            self.gurobi_model,
            self.feature_encoder.continuous_features,
            self.continuous_levels, self.continuous_vars,
            self.isolation_ensemble, self.isolation_flow_vars,
            self.isolation_continuous_left_constraints,
            self.isolation_continuous_right_constraints,
            self.isolation_continuous_logical_constraints,
            prefix="isolation_continuous"
        )

    def add_categorical_constraints(self):
        add_feature_categorical_constraints(
            self.gurobi_model,
            self.feature_encoder.categorical_features,
            self.feature_encoder.categories, self.categorical_vars,
            self.tree_ensemble, self.flow_vars,
            self.categorical_left_constraints,
            self.categorical_right_constraints,
            self.categorical_logical_constraints,
            prefix="tree_categorical"
        )
        add_feature_categorical_constraints(
            self.gurobi_model,
            self.feature_encoder.categorical_features,
            self.feature_encoder.categories, self.categorical_vars,
            self.isolation_ensemble, self.isolation_flow_vars,
            self.isolation_categorical_left_constraints,
            self.isolation_categorical_right_constraints,
            self.isolation_categorical_logical_constraints,
            prefix="isolation_categorical"
        )

    def add_feature_constraints(self):
        self.add_binary_constraints()
        self.add_discrete_constraints()
        self.add_continuous_constraints()
        self.add_categorical_constraints()

    def build_features(self):
        self.add_feature_vars()
        self.add_feature_constraints()

    def add_prob_vars(self):
        add_prob_vars(
            self.gurobi_model, self.prob_vars,
            list(range(self.tree_ensemble.n_classes)),
            name="prob"
        )

    def add_prob_constraints(self):
        add_prob_constraints(
            self.gurobi_model, self.tree_ensemble,
            list(range(self.tree_ensemble.n_classes)),
            self.flow_vars, self.prob_vars, self.weights,
            self.prob_constraints,
            prefix="prob"
        )

    def build_trees(self):
        add_tree_ensemble_vars_and_constraints(
            self.gurobi_model, self.tree_ensemble,
            self.flow_vars, self.branch_vars,
            self.root_constraints, self.flow_constraints,
            self.branch_to_left_constraints,
            self.branch_to_right_constraints,
            prefix="tree",
        )
        add_tree_ensemble_vars_and_constraints(
            self.gurobi_model, self.isolation_ensemble,
            self.isolation_flow_vars, self.isolation_branch_vars,
            self.isolation_root_constraints, self.isolation_flow_constraints,
            self.isolation_branch_to_left_constraints,
            self.isolation_branch_to_right_constraints,
            prefix="isolation",
        )

    def add_anomaly_var(self):
        self.anomaly_var = self.gurobi_model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            name="anomaly"
        )

    def add_isolation_plausibility_constraints(self):
        conss = add_isolation_plausibility_constraints(
            self.gurobi_model, self.isolation_ensemble,
            self.isolation_flow_vars, self.max_samples,
            self.anomaly_var, self.offset
        )
        self.isolation_constraint = conss[1]
        self.anomaly_constraint = conss[0]

    def init(self):
        self.gurobi_model = gp.Model("FIPEOracle")

        self.flow_vars = gp.tupledict()
        self.branch_vars = gp.tupledict()
        self.root_constraints = gp.tupledict()
        self.flow_constraints = gp.tupledict()
        self.branch_to_left_constraints = gp.tupledict()
        self.branch_to_right_constraints = gp.tupledict()

        self.isolation_flow_vars = gp.tupledict()
        self.isolation_branch_vars = gp.tupledict()
        self.isolation_root_constraints = gp.tupledict()
        self.isolation_flow_constraints = gp.tupledict()
        self.isolation_branch_to_left_constraints = gp.tupledict()
        self.isolation_branch_to_right_constraints = gp.tupledict()

        self.binary_vars = gp.tupledict()
        self.discrete_vars = gp.tupledict()
        self.continuous_vars = gp.tupledict()
        self.categorical_vars = gp.tupledict()

        self.binary_left_constraints = gp.tupledict()
        self.binary_right_constraints = gp.tupledict()
        self.isolation_binary_left_constraints = gp.tupledict()
        self.isolation_binary_right_constraints = gp.tupledict()

        self.discrete_left_constraints = gp.tupledict()
        self.discrete_right_constraints = gp.tupledict()
        self.discrete_logical_constraints = gp.tupledict()
        self.isolation_discrete_left_constraints = gp.tupledict()
        self.isolation_discrete_right_constraints = gp.tupledict()
        self.isolation_discrete_logical_constraints = gp.tupledict()

        self.continuous_left_constraints = gp.tupledict()
        self.continuous_right_constraints = gp.tupledict()
        self.continuous_logical_constraints = gp.tupledict()
        self.isolation_continuous_left_constraints = gp.tupledict()
        self.isolation_continuous_right_constraints = gp.tupledict()
        self.isolation_continuous_logical_constraints = gp.tupledict()

        self.categorical_left_constraints = gp.tupledict()
        self.categorical_right_constraints = gp.tupledict()
        self.categorical_logical_constraints = gp.tupledict()
        self.isolation_categorical_left_constraints = gp.tupledict()
        self.isolation_categorical_right_constraints = gp.tupledict()
        self.isolation_categorical_logical_constraints = gp.tupledict()

        self.prob_vars = gp.tupledict()
        self.prob_constraints = gp.tupledict()
        self.active_prob_vars = gp.tupledict()
        self.active_prob_constraints = gp.tupledict()
        self.majority_class_constraints = gp.tupledict()

    def build(self):
        self.init()

        self.build_trees()
        self.build_features()

        self.add_prob_vars()
        self.add_prob_constraints()
        self.add_anomaly_var()
        self.add_isolation_plausibility_constraints()

    def add_active_prob_vars(self, classes: list[int]):
        add_prob_vars(
            self.gurobi_model, self.active_prob_vars,
            classes, name="active_prob"
        )

    def add_active_prob_constraints(self, classes: int | list[int]):
        active = self.active
        add_prob_constraints(
            self.gurobi_model, self.tree_ensemble,
            classes, self.flow_vars, self.active_prob_vars,
            self.weights * active,
            self.active_prob_constraints,
            prefix="active_prob"
        )

    def add_majority_class_constraints(self, mc: int):
        wm = self.weights.min()
        eps = self.eps
        add_majority_class(
            self.gurobi_model,
            self.active_prob_vars, mc, wm * eps,
            self.majority_class_constraints,
            name="majority_class"
        )

    def add_objective(self):
        add_prob_objective(
            self.gurobi_model,
            self.active_prob_vars,
            self.c1, self.c2
        )

    def add(self, mc: int, c: int):
        self.c1 = mc
        self.c2 = c
        self.add_active_prob_vars([mc, c])
        self.add_active_prob_constraints([mc, c])
        self.add_majority_class_constraints(mc)
        self.add_objective()

    def reset(self):
        self.gurobi_model.remove(self.active_prob_vars)
        self.gurobi_model.remove(self.active_prob_constraints)
        self.gurobi_model.remove(self.majority_class_constraints)

    def optimize(self):
        c1 = self.c1
        c2 = self.c2
        eps = self.eps
        wm = self.weights.min()

        def callback(model: gp.Model, where: int):
            if where == GRB.Callback.MIPSOL:
                cutoff = (wm * eps if c1 < c2 else 0.0)
                val = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                if val < cutoff:
                    model.terminate()

        self.gurobi_model.optimize(callback)

    def set_gurobi_parameter(self, param, value):
        self.gurobi_model.setParam(param, value)

    def add_active(self, active):
        self.active = np.asarray(active)

    def separate(self, active):
        logger.debug("Separating the classes.")
        X = []
        nc = self.tree_ensemble.n_classes
        self.add_active(active)
        for c1 in range(nc):
            for c2 in range(nc):
                if c1 == c2:
                    continue

                self.add(c1, c2)
                self.optimize()

                if self.gurobi_model.SolCount == 0:
                    msg = ("When solving the FIPE Oracle"
                           " problem, no solution was found.")
                    logger.warning(msg)
                    continue

                Xi = self.get_all()

                X.extend(Xi)
                self.reset()
        columns = self.feature_encoder.columns
        return to_array_values(columns, X)

    def get_feature_values(self):
        return get_feature_values(
            self.feature_encoder.binary_features,
            self.binary_vars,
            self.feature_encoder.discrete_features,
            self.feature_encoder.values,
            self.discrete_vars,
            self.feature_encoder.continuous_features,
            self.continuous_levels,
            self.continuous_vars,
            self.feature_encoder.categorical_features,
            self.feature_encoder.categories,
            self.categorical_vars
        )

    def get_all(self, check=True):
        X = []
        c1 = self.c1
        c2 = self.c2
        wm = self.weights.min()
        eps = self.eps
        cutoff = (wm * eps if c1 < c2 else 0.0)
        for i in range(self.gurobi_model.SolCount):
            self.set_gurobi_parameter("SolutionNumber", i)
            if check:
                self.check_solution()

            obj = self.gurobi_model.PoolObjVal
            if obj < cutoff:
                continue

            x = self.get_feature_values()
            X.append(x)
        return X

    # TODO: this function should be added
    # TODO: as a test method in the future.
    def check_solution(self):
        x = self.get_feature_values()
        xa = to_array_values(self.feature_encoder.columns, x).values

        # Check if the path in the tree is correct.
        tree_ensemble = self.tree_ensemble
        E = tree_ensemble.ensemble_model

        for t, tree in enumerate(tree_ensemble):
            e = E[t]
            path = e.decision_path(xa)

            n = tree.root
            while n not in tree.leaves:
                f = tree.feature[n]
                left, right = tree.left[n], tree.right[n]

                to_left = self.flow_vars[t, left].Xn > 0.5
                mip_next = (left if to_left else right)

                to_left = path[0, left] > 0.5
                path_next = (left if to_left else right)

                if mip_next != path_next:
                    msg = (
                        "The path in the tree is incorrect."
                        f"The tree is: {t}"
                        f"The node is: {n}"
                        f"The feature is: {f}"
                        f"The MIP path is: {mip_next}"
                        f"The decision path is: {path_next}"
                    )
                    logger.debug(msg)
                    break

                n = mip_next

        # Check if the probabilities are correct.
        nc = tree_ensemble.n_classes
        p = predict_proba(E, xa, self.weights)
        p_mip = np.array([self.prob_vars[c].Xn for c in range(nc)])
        p_mip /= p_mip.sum()
        if not np.isclose(p, p_mip).all():
            msg = "The probabilities are incorrect."
            logger.debug(msg)

            msg = f"Predict: {p}"
            logger.debug(msg)
            msg = f"MIP: {p_mip}"
            logger.debug(msg)

        # Check if the pruning probabilities are correct.
        p = predict_proba(E, xa, self.weights * self.active)
        m = len(E)
        c1 = self.c1
        p1 = self.active_prob_vars[c1].Xn / m
        if not np.isclose(p[0, c1], p1):
            msg = f"The pruning probabilities are incorrect for class {c1}."
            logger.debug(msg)

            msg = f"Predict: {p[0, c1]}"
            logger.debug(msg)

            msg = f"MIP: {p1}"
            logger.debug(msg)

        c2 = self.c2
        p2 = self.active_prob_vars[c2].Xn / m
        if not np.isclose(p[0, c2], p2):
            msg = f"The pruning probabilities are incorrect for class {c2}."
            logger.debug(msg)

            msg = f"Predict: {p[0, c2]}"
            logger.debug(msg)

            msg = f"MIP: {p2}"
            logger.debug(msg)
