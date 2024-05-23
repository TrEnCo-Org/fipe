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


class OCEANI(
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
        self._build_features()
        self._build_ensemble()
        self._build_isolation()

    def _build_features(self):
        self.feature_vars = build_feature_vars(
            model=self.model,
            features=self.features,
            levels=self.levels,
            categories=self.categories
        )

    def _build_ensemble(self):
        for t, tree in enumerate(self.ensemble):
            self.flow_vars[t] = build_flow_vars(
                model=self.model,
                tree=tree,
                feature_vars=self.feature_vars,
                name=f"tree_{t}"
            )

    def _build_isolation(self):
        if self.isolation_ensemble is None:
            return
        for t, tree in enumerate(self.isolation_ensemble):
            self.isolation_flow_vars[t] = build_flow_vars(
                model=self.model,
                tree=tree,
                feature_vars=self.feature_vars,
                name=f"isolation_tree_{t}"
            )
        self._add_isolation_constr()

    def _add_isolation_constr(self):
        if self.isolation_ensemble is None:
            return
        else:
            n_estimators = self.isolation_ensemble.n_estimators
            max_samples = self.isolation_ensemble.max_samples
            average_depth = _average_path_length([max_samples])[0]
            lhs = gp.quicksum(
                self.isolation_flow_vars[t].weighted_depth
                for t in range(n_estimators)
            ) / (average_depth * n_estimators)
            min_score = -np.log2(-self.isolation_ensemble.offset)
            self.model.addConstr(
                lhs >= min_score,
                name="isolation_plausibility"
            )


class OCEANII(OCEANI):
    _prob_vars: gp.tupledict[int, gp.Var]
    _prob_constrs: gp.tupledict[int, gp.Constr]

    _majority_class_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        features: Features,
        ensemble: Ensemble,
        weights,
        isolation_ensemble: IsolationEnsemble | None = None,
        **kwargs
    ):
        OCEANI.__init__(
            self,
            features,
            ensemble,
            weights,
            isolation_ensemble,
            **kwargs
        )

    def build(self):
        OCEANI.build(self)
        self._build_prob_vars()

    def set_majority_class(self, c: int):
        self._add_majority_class_constrs(c)

    def clear_majority_class(self):
        self._remove_majority_class_constrs()

    def _build_prob_vars(self):
        vars, constrs = build_weighted_prob_vars(
            model=self.model,
            flow_vars=self.flow_vars,
            classes=list(range(self.n_classes)),
            weights=self.weights,
        )
        self._prob_vars = vars
        self._prob_constrs = constrs

    def _add_majority_class_constrs(self, c: int):
        self._majority_class_constrs = gp.tupledict()
        for k in range(self.n_classes):
            if k == c:
                continue
            self._add_majority_class_constr(c, k)

    def _add_majority_class_constr(self, c: int, k: int):
        mw = self.min_weight
        eps = self.eps
        rhs = (0.0 if c < k else eps * mw)

        constr = self.model.addConstr(
            self._prob_vars[c]
            >=
            self._prob_vars[k] + rhs,
            name=f"majority_class_{c}_{k}"
        )
        self._majority_class_constrs[k] = constr

    def _remove_majority_class_constrs(self):
        self.model.remove(self._majority_class_constrs)
        self._majority_class_constrs = gp.tupledict()


class OCEANIII(OCEANII):
    _activated_weights: dict[int, float]
    _activated_prob_vars: gp.tupledict[int, gp.Var]
    _activated_prob_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        features: Features,
        ensemble: Ensemble,
        weights,
        isolation_ensemble: IsolationEnsemble | None = None,
        **kwargs
    ):
        OCEANII.__init__(
            self,
            features,
            ensemble,
            weights,
            isolation_ensemble,
            **kwargs
        )

    def _set_activated(self, activated_weights):
        self._activated_weights = dict()
        for t in range(self.n_estimators):
            try:
                self._activated_weights[t] = activated_weights[t]
            except Exception:
                continue

    def _build_activated_prob_vars(self):
        vars, constrs = build_weighted_prob_vars(
            model=self.model,
            flow_vars=self.flow_vars,
            classes=list(range(self.n_classes)),
            weights=self._activated_weights,
            name="activated"
        )
        self._activated_prob_vars = vars
        self._activated_prob_constrs = constrs

    def _remove_activated(self):
        self.model.remove(self._activated_prob_constrs)
        self.model.remove(self._activated_prob_vars)
        self._activated_prob_vars = gp.tupledict()
        self._activated_prob_constrs = gp.tupledict()


class OCEANIV(OCEANIII):
    def __init__(
        self,
        features: Features,
        ensemble: Ensemble,
        weights,
        isolation_ensemble: IsolationEnsemble | None = None,
        **kwargs
    ):
        OCEANIII.__init__(
            self,
            features,
            ensemble,
            weights,
            isolation_ensemble,
            **kwargs
        )

    def add_objective(self, obj, sense):
        self.model.setObjective(obj, sense=sense)

    def get_counters(
        self,
        cutoff: float | str = 'all',
        check: bool = True
    ):
        param = GRB.Param.SolutionNumber
        for i in range(self.model.SolCount):
            self.set_gurobi_param(param, i)
            counter = self.feature_vars.value
            if check:
                self._check_counter(counter)
            if not (cutoff == 'all'):
                assert isinstance(cutoff, float)
                if (
                    self.model.ModelSense == GRB.MAXIMIZE
                    and self.model.PoolObjVal < cutoff
                ):
                    continue
                if (
                    self.model.ModelSense == GRB.MINIMIZE
                    and self.model.PoolObjVal > cutoff
                ):
                    continue
            yield counter
        self.set_gurobi_param(param, 0)

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
            self._prob_vars[c].Xn
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
            self._activated_prob_vars[c].Xn
            for c in range(self.n_classes)
        ])
        activated_prob = activated_prob / activated_prob.sum()
        w = np.array([
            self._activated_weights[t]
            for t in range(self.n_estimators)
        ])
        expected_activated_prob = predict_proba(self.ensemble.base, X, w)

        if not np.allclose(activated_prob, expected_activated_prob):
            msg = (f"The activated probabilities do not match.\n"
                   f" Expected: {expected_activated_prob}.\n"
                   f" Found: {activated_prob}.")
            raise ValueError(msg)
