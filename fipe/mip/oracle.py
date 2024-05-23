import numpy as np
import gurobipy as gp
from gurobipy import GRB

from sklearn.ensemble._iforest import _average_path_length  # type: ignore

from .abc import (
    MIP,
    EPS,
    WeightedModel,
    FeatureContainer,
    EnsembleContainer
)
from .flow import FlowVars
from .feature import FeatureVars
from ..typing import numeric, Sample
from ..ensemble import (
    Tree,
    Ensemble,
    IsolationEnsemble,
    Features,
    EnsembleParser,
    predict_proba
)


def build_feature_vars(
    model: gp.Model,
    features: Features,
    levels: dict[str, list[numeric]],
    categories: dict[str, list[str]]
) -> FeatureVars:
    feature_vars = FeatureVars()
    for f in features.binary:
        feature_vars.add_binary(f)
    for f in features.continuous:
        feature_vars.add_continuous(f, levels[f])
    for f in features.categorical:
        feature_vars.add_categorical(f, categories[f])
    feature_vars.build_all(model)
    return feature_vars


def build_flow_vars(
    model: gp.Model,
    tree: Tree,
    feature_vars: FeatureVars,
    name: str = ""
) -> FlowVars:
    flow_vars = FlowVars(tree=tree, name=name)
    flow_vars.build(model=model)
    flow_vars.add_feature_constrs(
        model=model,
        feature_vars=feature_vars
    )
    return flow_vars


def build_weighted_prob_vars(
    model: gp.Model,
    flow_vars: dict[int, FlowVars],
    classes: list[int],
    weights: dict[int, float],
    name: str = ""
) -> tuple[
    gp.tupledict[int, gp.Var],
    gp.tupledict[int, gp.Constr]
]:
    vars = gp.tupledict()
    constrs = gp.tupledict()
    for c in classes:
        vars[c] = model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            name=f"{name}_prob_{c}"
        )
        constrs[c] = model.addConstr(
            vars[c] == gp.quicksum(
                weights[t]
                * flow_vars[t].prob[c]
                for t in weights
            ),
            name=f"{name}_prob_{c}"
        )
    return vars, constrs


class OCEAN(
    MIP,
    EPS,
    WeightedModel,
    EnsembleParser,
    FeatureContainer,
    EnsembleContainer
):
    isolation_ensemble: IsolationEnsemble | None

    feature_vars: FeatureVars
    flow_vars: dict[int, FlowVars]
    isolation_flow_vars: dict[int, FlowVars]

    prob_vars: gp.tupledict[int, gp.Var]
    prob_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        features: Features,
        ensemble: Ensemble,
        weights,
        isolation_ensemble: IsolationEnsemble | None = None,
        **kwargs
    ):
        MIP.__init__(self)
        EPS.__init__(self, **kwargs)
        WeightedModel.__init__(self, weights)
        EnsembleParser.__init__(self, **kwargs)
        FeatureContainer.__init__(self, features)
        EnsembleContainer.__init__(self, ensemble)

        self.isolation_ensemble = isolation_ensemble
        ensembles = [ensemble]
        if isolation_ensemble is not None:
            ensembles.append(isolation_ensemble)
        self.parse_levels(ensembles, features)

        self.flow_vars = dict()
        self.isolation_flow_vars = dict()

    def build(self):
        MIP.build(self, name="Oracle")
        self.build_features()
        self.build_ensemble()
        self.build_prob_vars()
        self.build_isolation()

    def build_features(self):
        self.feature_vars = build_feature_vars(
            model=self.model,
            features=self.features,
            levels=self.levels,
            categories=self.categories
        )

    def build_ensemble(self):
        for t, tree in enumerate(self.ensemble):
            self.flow_vars[t] = build_flow_vars(
                model=self.model,
                tree=tree,
                feature_vars=self.feature_vars,
                name=f"tree_{t}"
            )

    def build_prob_vars(self):
        vars, constrs = build_weighted_prob_vars(
            model=self.model,
            flow_vars=self.flow_vars,
            classes=list(range(self.n_classes)),
            weights=self.weights,
        )
        self.prob_vars = vars
        self.prob_constrs = constrs

    def build_isolation(self):
        if self.isolation_ensemble is None:
            return
        for t, tree in enumerate(self.isolation_ensemble):
            self.isolation_flow_vars[t] = build_flow_vars(
                model=self.model,
                tree=tree,
                feature_vars=self.feature_vars,
                name=f"isolation_tree_{t}"
            )
        self.add_isolation_constr()

    def add_isolation_constr(self):
        if self.isolation_ensemble is None:
            return
        else:
            average_depth = _average_path_length(
                [self.isolation_ensemble.max_samples]
            )[0] * self.n_estimators
            lhs = gp.quicksum(
                self.isolation_flow_vars[t].weighted_depth
                for t in self.isolation_flow_vars
            ) / average_depth
            min_score = -np.log2(-self.isolation_ensemble.offset)
            self.model.addConstr(
                lhs >= min_score,
                name="isolation_plausibility"
            )


class Oracle(OCEAN):
    activated_weights: dict[int, float]
    activated_prob_vars: gp.tupledict[int, gp.Var]
    activated_prob_constrs: gp.tupledict[int, gp.Constr]

    majority_class_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        features: Features,
        ensemble: Ensemble,
        weights,
        isolation_ensemble: IsolationEnsemble | None = None,
        **kwargs
    ):
        OCEAN.__init__(
            self,
            features,
            ensemble,
            weights,
            isolation_ensemble,
            **kwargs
        )

    def separate(self, activated_weights):
        self._set_activated(activated_weights)
        self._add_activated_prob_vars()
        for c in range(self.n_classes):
            self._add_majority_class_constrs(c)
            for counter in self._separate_single(c):
                yield counter
            self._clear_majority_class_constrs()
        self._clear_activated()

    def _separate_single(self, c: int):
        for k in range(self.n_classes):
            if c == k:
                continue
            self._separate_pair(c, k)
            for counter in self._get_counters(c, k):
                yield counter

    def _separate_pair(self, c1: int, c2: int):
        self._add_objective(c1, c2)
        callback = self._get_callback(c1, c2)
        self.model.optimize(callback)

    def _get_callback(self, c1: int, c2: int):
        mw = self.min_weight
        eps = self.eps

        def callback(model: gp.Model, where: int):
            if where == gp.GRB.Callback.MIPSOL:
                cutoff = (mw * eps if c1 < c2 else 0.0)
                val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                if val < cutoff:
                    model.terminate()
        return callback

    def _get_counters(
        self,
        c1: int,
        c2: int,
        check: bool = True
    ):
        mw = self.min_weight
        eps = self.eps
        cutoff = (mw * eps if c1 < c2 else 0.0)
        param = GRB.Param.SolutionNumber
        sol_count = self.model.SolCount
        if sol_count > 0:
            for i in range(sol_count):
                self.set_gurobi_param(param, i)
                counter = self.feature_vars.value
                if check:
                    self._check_counter(counter)
                if self.model.PoolObjVal < cutoff:
                    continue
                yield counter

    def _add_objective(self, c1: int, c2: int):
        vars = self.activated_prob_vars
        self.model.setObjective(
            vars[c2] - vars[c1],
            sense=gp.GRB.MAXIMIZE
        )

    def _set_activated(self, activated_weights):
        self.activated_weights = dict()
        for t in range(self.n_estimators):
            try:
                self.activated_weights[t] = activated_weights[t]
            except Exception:
                continue

    def _add_activated_prob_vars(self):
        vars, constrs = build_weighted_prob_vars(
            model=self.model,
            flow_vars=self.flow_vars,
            classes=list(range(self.n_classes)),
            weights=self.activated_weights,
            name="activated"
        )
        self.activated_prob_vars = vars
        self.activated_prob_constrs = constrs

    def _clear_activated(self):
        self.model.remove(self.activated_prob_constrs)
        self.model.remove(self.activated_prob_vars)
        self.activated_prob_vars = gp.tupledict()
        self.activated_prob_constrs = gp.tupledict()

    def _add_majority_class_constrs(self, c: int):
        self.majority_class_constrs = gp.tupledict()
        for k in range(self.n_classes):
            if k == c:
                continue
            self._add_majority_class_constr(c, k)

    def _clear_majority_class_constrs(self):
        self.model.remove(self.majority_class_constrs)
        self.majority_class_constrs = gp.tupledict()

    def _add_majority_class_constr(self, c: int, k: int):
        mw = self.min_weight
        eps = self.eps
        rhs = (0.0 if c < k else eps * mw)

        constr = self.model.addConstr(
            self.prob_vars[c]
            >=
            self.prob_vars[k] + rhs,
            name=f"majority_class_{c}_{k}"
        )
        self.majority_class_constrs[k] = constr

    def _check_counter(self, x: Sample):
        X = self.transform(x)
        # Check if the path is valid for each tree
        for t in range(self.n_estimators):
            e = self.ensemble.base[t]
            mip_path = self.flow_vars[t].path
            tree_path = e.decision_path(X)
            tree = self.ensemble[t]
            node = tree.root
            while node in tree.internal_nodes:
                left = tree.left[node]
                right = tree.right[node]
                mip_next = left if mip_path[left] else right
                tree_next = left if tree_path[0, left] else right
                if mip_next != tree_next:
                    msg = (f"The MIP path and the tree path"
                           f" for tree {t} of the ensemble"
                           f" diverge at node {node}.\n"
                           f" Threshold: {tree.threshold[node]}.\n"
                           f" Feature: {tree.feature[node]}.\n"
                           f" Value: {x[tree.feature[node]]}.\n"
                           f" MIP path: {mip_path}.\n"
                           f" Tree path: {tree_path}.")
                    raise ValueError(msg)

                node = mip_next
        # Check if the probabilities are the same
        prob = np.array([
            self.prob_vars[c].Xn
            for c in range(self.n_classes)
        ])
        prob = prob / prob.sum()
        w = np.array([
            self.weights[t]
            for t in range(self.n_estimators)
        ])
        expected_prob = predict_proba(self.ensemble.base, X, w)
        if not np.allclose(prob, expected_prob):
            msg = (f"The probabilities do not match.\n"
                   f" Expected: {expected_prob}.\n"
                   f" Found: {prob}.")
            raise ValueError(msg)
        # Check if the activated probabilities are the same
        activated_prob = np.array([
            self.activated_prob_vars[c].Xn
            for c in range(self.n_classes)
        ])
        activated_prob = activated_prob / activated_prob.sum()
        w = np.array([
            self.activated_weights[t]
            for t in range(self.n_estimators)
        ])
        expected_activated_prob = predict_proba(self.ensemble.base, X, w)

        if not np.allclose(
            activated_prob,
            expected_activated_prob
        ):
            msg = (f"The activated probabilities do not match.\n"
                   f" Expected: {expected_activated_prob}.\n"
                   f" Found: {activated_prob}.")
            raise ValueError(msg)
