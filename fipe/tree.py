from collections.abc import Iterable, Iterator

import numpy as np
import warnings

from .encoding import FeatureEncoder

Node = int

class Tree(Iterable[Node]):
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
    
    def __init__(
        self,
        tree,
        feature_encoder: FeatureEncoder
    ):
        self.tree = tree
        
        self.root = 0
        self.n_nodes = tree.node_count
        self.max_depth = tree.max_depth
        self.internal_nodes = set()
        self.leaves = set()
        self.node_depth = dict()
        self.left = dict()
        self.right = dict()
        self.feature = dict()
        self.threshold = dict()
        self.category = dict()
        self.prob = dict()
        
        self.parse_tree(tree, feature_encoder)

    def nodes_at_depth(
        self,
        d: int,
        with_leaves: bool = False
    ) -> Iterator[Node]:
        return iter(
            n for n, depth in self.node_depth.items()
            if (depth == d and
                (with_leaves
                 or n in self.internal_nodes)
            )
        )

    def node_split_on(
        self,
        feature: str
    ) -> set[Node]:
        return set(n for n in self.internal_nodes
                   if self.feature[n] == feature)

    def __iter__(self) -> Iterator[Node]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    def parse_tree(
        self,
        tree,
        feature_encoder: FeatureEncoder
    ):
        def dfs(n, d):
            self.node_depth[n] = d
            l = tree.children_left[n]
            r = tree.children_right[n]
            if l == r:
                self.leaves.add(n)
                v = tree.value[n].flatten()
                p = v / v.sum()
                # This is for hard voting.
                q = np.argmax(p)
                self.prob[n] = np.eye(p.shape[0])[q]
                return
            else:
                f = tree.feature[n]
                f: str = feature_encoder.columns[f]
                
                if f in feature_encoder.inverse_categories:
                    self.category[n] = f
                    f = feature_encoder.inverse_categories[f]

                self.feature[n] = f
                
                if f in feature_encoder.numerical_features:
                    self.threshold[n] = tree.threshold[n]

                self.left[n], self.right[n] = l, r
                self.internal_nodes.add(n)
                dfs(l, d + 1)
                dfs(r, d + 1)
        dfs(self.root, 0)

class TreeEnsemble(Iterable[Tree]):
    n_trees: int
    n_classes: int
    trees: list[Tree]
    numerical_levels: dict[str, list[float]]
    tol: float

    def __init__(
        self,
        ensemble,
        feature_encoder: FeatureEncoder,
        **kwargs
    ):
        self.n_trees = len(ensemble)
        self.n_classes = ensemble[0].n_classes_
        self.trees = [
            Tree(tree.tree_, feature_encoder)
            for tree in ensemble
        ]
        self.numerical_levels = dict()
        self.tol = kwargs.get("tol", 1e-4)
        
        self.parse_numerical_levels(feature_encoder)

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
                warnings.warn(
                    f"The feature {f} is not used in any split."
                )
            levels = list(sorted(levels))
            if np.diff(levels).min() < self.tol:
                warnings.warn(
                    f"The levels of the feature {f} are too close to each other."
                )
            self.numerical_levels[f] = list(sorted(levels))
