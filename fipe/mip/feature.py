from abc import ABC, abstractmethod
from itertools import chain

import numpy as np
import gurobipy as gp

from ..typing import numeric, Sample


class FeatureVar(ABC):
    name: str

    def __init__(self, name: str = ""):
        self.name = name

    @abstractmethod
    def build(self, model: gp.Model):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError("Subclasses must implement this method")


class BinaryVar(FeatureVar):
    var: gp.Var

    def __init__(self, name: str = ""):
        super().__init__(name)

    def build(self, model: gp.Model):
        self._add_var(model)

    @property
    def value(self):
        return self.var.Xn

    def _add_var(self, model: gp.Model):
        self.var = model.addVar(
            vtype=gp.GRB.BINARY,
            name=self.name
        )


class ContinuousVar(
    FeatureVar,
    gp.tupledict[int, gp.Var]
):
    levels: list[numeric]
    _logic_constrs: gp.tupledict[int, gp.Constr]

    def __init__(
        self,
        levels: list[numeric],
        name: str = ""
    ):
        FeatureVar.__init__(self, name)
        gp.tupledict.__init__(self)
        self.levels = levels
        self._logic_constrs = gp.tupledict()

    def build(self, model: gp.Model):
        self._add_vars(model)
        self._add_logic_constrs(model)

    @property
    def value(self) -> numeric:
        n = len(self.levels)
        mu = [self[j].Xn for j in range(n)]
        mu = np.array(mu)
        nu = -np.diff(mu)
        levels = np.array(self.levels)
        x = (levels[:-1] + levels[1:]) / 2
        return np.dot(nu, x)

    def __getitem__(self, j: int) -> gp.Var:
        return gp.tupledict.__getitem__(self, j)

    def __setitem__(self, j: int, var: gp.Var):
        gp.tupledict.__setitem__(self, j, var)

    def _add_vars(self, model: gp.Model):
        n = len(self.levels)
        for j in range(n):
            self[j] = model.addVar(
                vtype=gp.GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{self.name}_{j}"
            )

    def _add_logic_constrs(self, model: gp.Model):
        n = len(self.levels)
        for j in range(n):
            if j == 0:
                expr = (self[j] == 1.0)
            elif j == n - 1:
                expr = (self[j] == 0.0)
            else:
                expr = (self[j-1] >= self[j])

            self._logic_constrs[j] = model.addConstr(
                expr,
                name=f"{self.name}_logic_{j}"
            )


class CategoricalVar(
    FeatureVar,
    gp.tupledict[str, gp.Var]
):
    categories: list[str]

    _logic_constr: gp.Constr

    def __init__(
        self,
        categories: list[str],
        name: str = ""
    ):
        FeatureVar.__init__(self, name)
        gp.tupledict.__init__(self)
        self.categories = categories

    def build(self, model: gp.Model):
        self._add_vars(model)
        self._add_logic_constr(model)

    @property
    def value(self):
        return {
            cat: self[cat].Xn
            for cat in self.categories
        }

    def __setitem__(self, cat: str, var: gp.Var):
        gp.tupledict.__setitem__(self, cat, var)

    def __getitem__(self, cat: str) -> gp.Var:
        return gp.tupledict.__getitem__(self, cat)

    def _add_vars(self, model: gp.Model):
        for cat in self.categories:
            self[cat] = model.addVar(
                vtype=gp.GRB.BINARY,
                name=f"{self.name}_{cat}"
            )

    def _add_logic_constr(self, model: gp.Model):
        self._logic_constr = model.addConstr(
            gp.quicksum(
                self[cat]
                for cat in self.categories
            ) == 1.0,
            name=f"{self.name}_logic"
        )


class FeatureVars:
    levels: dict[str, list[numeric]]
    categories: dict[str, list[str]]

    binary: dict[str, BinaryVar]
    continuous: dict[str, ContinuousVar]
    categorical: dict[str, CategoricalVar]

    def __init__(self):
        self.levels = dict()
        self.categories = dict()
        self.binary = dict()
        self.continuous = dict()
        self.categorical = dict()

    def build_all(self, model: gp.Model):
        for var in chain(
            self.binary.values(),
            self.continuous.values(),
            self.categorical.values()
        ):
            var.build(model)

    def add_binary(self, feature: str):
        var = BinaryVar(feature)
        self.binary[feature] = var

    def add_continuous(
        self,
        feature: str,
        levels: list[numeric]
    ):
        self.levels[feature] = levels
        var = ContinuousVar(levels, feature)
        self.continuous[feature] = var

    def add_categorical(
        self,
        feature: str,
        categories: list[str]
    ):
        self.categories[feature] = categories
        var = CategoricalVar(categories, feature)
        self.categorical[feature] = var

    @property
    def value(self) -> Sample:
        v = dict()
        for f, var in chain(
            self.binary.items(),
            self.continuous.items(),
        ):
            v[f] = var.value
        for f, var in self.categorical.items():
            v.update(var.value)
        return v
