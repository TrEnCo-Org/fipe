from collections.abc import Iterable, Iterator

import numpy as np

from sklearn.tree import _tree

from .encoding import FeatureEncoder

import warnings


Node = int


class Tree(Iterable[Node]):
    tree: _tree.Tree
    root: Node
    n_nodes: int
    max_depth: int
    internal_nodes: set[Node]
    leaves: set[Node]
    node_depth: dict[Node, int]
    left: dict[Node, Node]
    right: dict[Node, Node]
    feature: dict[Node, str]
    threshold: dict[Node, float]
    category: dict[Node, str]
    prob: dict[Node, np.ndarray]
    n_samples: dict[Node, int]

    def __init__(
        self,
        tree: _tree.Tree,
        feature_encoder: FeatureEncoder
    ):
        self.tree = tree

        self.root = 0
        self.n_nodes = tree.node_count
        self.max_depth = tree.max_depth  # type: ignore
        self.internal_nodes = set()
        self.leaves = set()
        self.node_depth = dict()
        self.left = dict()
        self.right = dict()
        self.feature = dict()
        self.threshold = dict()
        self.category = dict()
        self.prob = dict()
        self.n_samples = dict()

        self.parse_tree(tree, feature_encoder)

    def nodes_at_depth(
        self,
        d: int,
        with_leaves: bool = False
    ) -> set[Node]:
        def fn(n):
            return self.node_depth[n] == d
        nodes = (self if with_leaves else self.internal_nodes)
        return set(filter(fn, nodes))

    def node_split_on(
        self,
        feature: str
    ) -> set[Node]:
        def fn(n):
            return self.feature[n] == feature
        return set(filter(fn, self.internal_nodes))

    def __iter__(self) -> Iterator[Node]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    def parse_tree(
        self,
        tree,
        feature_encoder: FeatureEncoder
    ):
        def dfs(node, depth):
            self.node_depth[node] = depth
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:
                self.leaves.add(node)
                v = tree.value[node].flatten()
                p = v / v.sum()
                # This is for hard voting.
                q = np.argmax(p)
                k = p.shape[0]
                self.prob[node] = np.eye(k)[q]
                self.n_samples[node] = tree.n_node_samples[node]
                return
            else:
                i: int = tree.feature[node]
                f: str = feature_encoder.columns[i]

                if f in feature_encoder.inverse_categories:
                    self.category[node] = f
                    f = feature_encoder.inverse_categories[f]

                self.feature[node] = f

                if f in feature_encoder.numerical_features:
                    self.threshold[node] = tree.threshold[node]

                self.left[node] = left
                self.right[node] = right
                self.internal_nodes.add(node)
                dfs(left, depth + 1)
                dfs(right, depth + 1)
        dfs(self.root, 0)


class TreeEnsemble(Iterable[Tree]):
    trees: list[Tree]
    numerical_levels: dict[str, list[float]]
    tol: float

    def __init__(
        self,
        ensemble_model,
        feature_encoder: FeatureEncoder,
        **kwargs
    ):
        self.ensemble_model = ensemble_model
        self.trees = [
            Tree(tree.tree_, feature_encoder)
            for tree in ensemble_model
        ]
        self.numerical_levels = dict()
        self.tol = kwargs.get("tol", 1e-4)

        self.parse_numerical_levels(feature_encoder)

    @property
    def n_trees(self) -> int:
        return len(self.ensemble_model)

    @property
    def n_classes(self) -> int:
        return self.ensemble_model[0].n_classes_

    def __iter__(self) -> Iterator[Tree]:
        return iter(self.trees)

    def __len__(self) -> int:
        return self.n_trees

    def __getitem__(self, t: int) -> Tree:
        return self.trees[t]

    def parse_numerical_levels(
        self,
        feature_encoder: FeatureEncoder
    ):
        for f in feature_encoder.continuous_features:
            levels = set()
            levels.add(feature_encoder.lower_bounds[f])
            for tree in self.trees:
                for n in tree.node_split_on(f):
                    levels.add(tree.threshold[n])
            levels.add(feature_encoder.upper_bounds[f])
            if len(levels) == 2:
                msg = (f"The feature {f} is not used in any split."
                       " It will be ignored.")
                warnings.warn(msg)

            levels = list(sorted(levels))
            if np.diff(levels).min() < self.tol:
                msg = (f"The levels of the feature {f}"
                       " are too close to each other.")
                warnings.warn(msg)
            self.numerical_levels[f] = list(sorted(levels))
