from collections.abc import Iterable, Iterator

from sklearn.ensemble._base import BaseEnsemble

from .features import Features
from .tree import Tree


class Ensemble(Iterable[Tree]):
    base: BaseEnsemble
    trees: list[Tree]

    def __init__(
        self,
        base: BaseEnsemble,
        features: Features,
        **kwargs
    ):
        self.base = base
        self.numerical_levels = dict()
        self.tol = kwargs.get("tol", 1e-4)
        self._parse_trees(features)

    @property
    def n_estimators(self) -> int:
        return len(self.trees)

    @property
    def n_classes(self) -> int:
        return self.base[0].n_classes_

    @property
    def max_depth(self) -> int:
        return max(tree.max_depth for tree in self.trees)

    def __iter__(self) -> Iterator[Tree]:
        return iter(self.trees)

    def __len__(self) -> int:
        return self.n_estimators

    def __getitem__(self, t: int) -> Tree:
        return self.trees[t]

    def _parse_trees(self, features: Features):
        def fn(tree):
            return Tree(tree.tree_, features)
        self.trees = list(map(fn, self.base))
