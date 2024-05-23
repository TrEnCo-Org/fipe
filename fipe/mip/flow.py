import numpy as np
import gurobipy as gp
from gurobipy import GRB

from sklearn.ensemble._iforest import _average_path_length  # type: ignore

from .feature import (
    FeatureVars,
    BinaryVar,
    ContinuousVar,
    CategoricalVar
)
from ..ensemble import (
    Tree,
    Node
)


class FlowVars(gp.tupledict[Node, gp.Var]):
    """
    This class represents the flow variables of a tree.
    Each node in the tree has a flow variable associated with it.

    Parameters:
    -----------
    tree: Tree
        The tree for which the flow variables are created.
    name: str
        The name of the flow variables.
    """
    name: str
    tree: Tree

    _flow_constrs: gp.tupledict[Node, gp.Constr]
    _depth_vars: gp.tupledict[int, gp.Var]
    _left_constrs: gp.tupledict[tuple[int, Node], gp.Constr]
    _right_constrs: gp.tupledict[tuple[int, Node], gp.Constr]
    _root_constr: gp.Constr

    def __init__(
        self,
        tree: Tree,
        name: str = ""
    ):
        gp.tupledict.__init__(self)
        self.tree = tree
        self.name = name
        self._flow_constrs = gp.tupledict()
        self._depth_vars = gp.tupledict()
        self._left_constrs = gp.tupledict()
        self._right_constrs = gp.tupledict()

    def build(self, model: gp.Model):
        """
        Build the flow variables and constraints.

        Parameters:
        -----------
        model: gp.Model
            The Gurobi model to which the variables
            and constraints are added.
        """
        self._add_flow_vars(model=model)
        self._add_depth_vars(model=model)
        self._add_root_constr(model=model)
        self._add_flow_constrs(model=model)
        self._add_branch_constrs(model=model)

    def add_branch_rule(
        self,
        model: gp.Model,
        var: gp.Var,
        node: Node,
        name: str = "branch"
    ):
        """
        Add a branching rule to flow variable of the tree
        based on the `var` variable at the given `node`.

        This means that if `var` is 1, then the flow variable
        at the left child of `node` is also 1, and the flow
        variable at the right child of `node` is 0.
        However, if `var` is 0, then the flow variable at the
        left child of `node` is 0, and the flow variable at the
        right child of `node` is 1.

        Parameters:
        -----------
        model: gp.Model
            The Gurobi model to which the constraints are added.
        var: gp.Var
            The variable based on which the branching rule is applied.
        node: Node
            The node in the tree to which the branching rule is applied.
        name: str
            The name of the branching rule.

        Returns:
        --------
        left_constr: gp.Constr
            The constraint that the flow variable at the left child
            of `node` is 1 if `var` is 1.
        right_constr: gp.Constr
            The constraint that the flow variable at the right child
            of `node` is 1 if `var` is 0.
        """
        left = self.tree.left[node]
        right = self.tree.right[node]
        left_constr = model.addConstr(
            self[left] <= 1 - var,
            name=f"{name}_left_{node}"
        )
        right_constr = model.addConstr(
            self[right] <= var,
            name=f"{name}_right_{node}"
        )
        return left_constr, right_constr

    def add_feature_constrs(
        self,
        model: gp.Model,
        feature_vars: FeatureVars
    ):
        """
        Add constraints to the flow variables based on the
        feature variables.
        """
        self._add_binary_constrs(model, feature_vars.binary)
        self._add_continuous_constrs(model, feature_vars.continuous)
        self._add_categorical_constrs(model, feature_vars.categorical)

    @property
    def prob(self):
        """
        Get the probability expression of the flow variables
        for each class in the tree.
        """
        return dict({
            c: gp.quicksum(
                self.tree.prob[c][node]
                * self[node]
                for node in self.tree.leaves
            )
            for c in self.tree.prob
        })

    @property
    def weighted_depth(self):
        def fn(n):
            return _average_path_length([n])[0]

        return gp.quicksum(
            fn(self.tree.n_samples[node])
            * self.tree.node_depth[node]
            * self[node]
            for node in self.tree.leaves
        )

    @property
    def path(self):
        return np.array([self[n].Xn >= 0.5 for n in self.tree])

    def __setitem__(self, node: Node, var: gp.Var):
        gp.tupledict.__setitem__(self, node, var)

    def __getitem__(self, node: Node) -> gp.Var:
        return gp.tupledict.__getitem__(self, node)

    def _add_flow_vars(self, model: gp.Model):
        for node in self.tree:
            self[node] = model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{self.name}_flow_{node}"
            )

    def _add_depth_vars(self, model: gp.Model):
        for depth in range(self.tree.max_depth):
            self._depth_vars[depth] = model.addVar(
                vtype=GRB.BINARY,
                name=f"{self.name}_depth_{depth}"
            )

    def _add_root_constr(self, model: gp.Model):
        self._root_constr = model.addConstr(
            self[self.tree.root] == 1.0,
            name=f"{self.name}_root"
        )

    def _add_flow_constrs(self, model: gp.Model):
        for node in self.tree.internal_nodes:
            left = self.tree.left[node]
            right = self.tree.right[node]
            self._flow_constrs[node] = model.addConstr(
                self[node] == self[left] + self[right],
                name=f"{self.name}_flow_{node}"
            )

    def _add_branch_constrs(self, model: gp.Model):
        for depth in range(self.tree.max_depth):
            var = self._depth_vars[depth]
            for node in self.tree.nodes_at_depth(depth):
                left, right = self.add_branch_rule(
                    model=model,
                    var=var,
                    node=node,
                    name=f"{self.name}_depth_{depth}"
                )
                self._left_constrs[depth, node] = left
                self._right_constrs[depth, node] = right

    def _add_binary_constrs(
        self,
        model: gp.Model,
        vars: dict[str, BinaryVar]
    ):
        for f, v in vars.items():
            for node in self.tree.nodes_split_on(f):
                self.add_branch_rule(
                    model=model,
                    var=v.var,
                    node=node,
                    name=f"{self.name}_binary_{f}"
                )

    def _add_continuous_constrs(
        self,
        model: gp.Model,
        vars: dict[str, ContinuousVar]
    ):
        for f, v in vars.items():
            for node in self.tree.nodes_split_on(f):
                th = self.tree.threshold[node]
                j = list(vars[f].levels).index(th)
                self.add_branch_rule(
                    model=model,
                    var=v[j],
                    node=node,
                    name=f"{self.name}_continuous_{f}"
                )

    def _add_categorical_constrs(
        self,
        model: gp.Model,
        vars: dict[str, CategoricalVar]
    ):
        for f, v in vars.items():
            for node in self.tree.nodes_split_on(f):
                cat = self.tree.category[node]
                self.add_branch_rule(
                    model=model,
                    var=v[cat],
                    node=node,
                    name=f"{self.name}_categorical_{cat}"
                )
