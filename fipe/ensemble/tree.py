from collections.abc import Iterable, Iterator
from collections import defaultdict

import numpy as np

from sklearn.tree._tree import Tree as _Tree

from .features import Features


# Node type is int
Node = int


class Tree(Iterable[Node]):
    """
    Class to represent a tree.

    This class is a wrapper around the sklearn.tree._tree.Tree

    Parameters:
    ------------
    tree: _Tree
        The tree to be represented.
    features: Features
        The features of the dataset.

    Attributes:
    ------------
    root: Node
        The root node of the tree.
    n_nodes: int
        The number of nodes in the tree.
    max_depth: int
        The maximum depth of the tree.
    internal_nodes: set[Node]
        The set of internal nodes in the tree.
    leaves: set[Node]
        The set of leaf nodes in the tree.
    node_depth: dict[Node, int]
        The depth of each node in the tree.
    left: dict[Node, Node]
        The left child of each internal node.
    right: dict[Node, Node]
        The right child of each internal node.
    feature: dict[Node, str]
        The feature split at each internal node.
    threshold: dict[Node, float]
        The threshold split at each internal node.
        This is only present for numerical features.
    category: dict[Node, str]
        The category split at each internal node.
        This is only present for categorical features.
    prob: defaultdict[int, dict[Node, float]]
        The probability of each class at each leaf node.
    """
    tree: _Tree
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
    prob: defaultdict[int, dict[Node, float]]
    n_samples: dict[Node, int]

    def __init__(
        self,
        tree: _Tree,
        features: Features
    ):
        self.tree = tree

        self.internal_nodes = set()
        self.leaves = set()
        self.node_depth = dict()
        self.left = dict()
        self.right = dict()
        self.feature = dict()
        self.threshold = dict()
        self.category = dict()
        self.prob = defaultdict(dict)
        self.n_samples = dict()

        self._parse_tree(tree, features)

    def nodes_at_depth(
        self,
        depth: int,
        with_leaves: bool = False
    ) -> set[Node]:
        """
        The set of nodes at a given depth.

        Parameters:
        ------------
        d: int
            The depth of the nodes.
        with_leaves: bool
            Whether to include leaf nodes.

        Returns:
        ---------
        set[Node]
            The set of nodes at the given depth.
        """
        def fn(n):
            return self.node_depth[n] == depth
        nodes = (self if with_leaves else self.internal_nodes)
        return set(filter(fn, nodes))

    def nodes_split_on(
        self,
        feature: str
    ) -> set[Node]:
        """
        The set of nodes that split on a given feature.

        Parameters:
        ------------
        feature: str
            The feature to split on.

        Returns:
        ---------
        set[Node]
            The set of nodes that split on the feature.
        """
        def fn(n):
            return self.feature[n] == feature
        return set(filter(fn, self.internal_nodes))

    def __iter__(self) -> Iterator[Node]:
        return iter(range(self.n_nodes))

    def __len__(self) -> int:
        return self.n_nodes

    def _parse_tree(
        self,
        tree,
        features: Features
    ):
        self.root = 0
        self.n_nodes = tree.node_count
        self.max_depth = tree.max_depth

        def dfs(node, depth):
            self.node_depth[node] = depth
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:
                self.leaves.add(node)
                v = tree.value[node].flatten()
                p = v / v.sum()

                # This part is for hard
                # voting classifiers
                # We need to adapt it for
                # soft voting classifiers
                # after experimenting with
                # the hard voting ones.
                q = np.argmax(p)
                k = p.shape[0]
                p = np.eye(k)[q]
                for c in range(k):
                    self.prob[c][node] = p[c]
                self.n_samples[node] = tree.n_node_samples[node]
                return
            else:
                i: int = tree.feature[node]
                f: str = features.columns[i]

                if f in features.inverse_categories:
                    self.category[node] = f
                    f = features.inverse_categories[f]

                self.feature[node] = f

                if f in features.numerical:
                    self.threshold[node] = tree.threshold[node]

                self.left[node] = left
                self.right[node] = right
                self.internal_nodes.add(node)
                dfs(left, depth + 1)
                dfs(right, depth + 1)
        dfs(self.root, 0)
