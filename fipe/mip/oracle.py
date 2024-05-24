import gurobipy as gp

from .oceans import OCEANIV
from ..ensemble import (
    Features,
    Ensemble,
    IsolationEnsemble
)


class Oracle(OCEANIV):
    def __init__(
        self,
        features: Features,
        ensemble: Ensemble,
        weights,
        isolation_ensemble: IsolationEnsemble | None = None,
        **kwargs
    ):
        OCEANIV.__init__(
            self,
            features,
            ensemble,
            weights,
            isolation_ensemble,
            **kwargs
        )

    def __call__(self, activated_weights):
        self._set_activated(activated_weights)
        self._build_activated_prob_vars()
        for c in range(self.n_classes):
            for counter in self._run_single(c):
                yield counter
        self._remove_activated()

    def _run_single(self, c: int):
        self.set_majority_class(c)
        for k in range(self.n_classes):
            if c == k:
                continue
            self._run_pair(c, k)
            for counter in self._get_pair_counters(c, k):
                yield counter
        self.clear_majority_class()

    def _run_pair(self, c1: int, c2: int):
        self._add_pair_objective(c1, c2)
        callback = self._get_pair_callback(c1, c2)
        self.optimize(callback)

    def _add_pair_objective(self, c1: int, c2: int):
        vars = self._activated_prob_vars
        obj = vars[c2] - vars[c1]
        self.set_objective(obj, sense=gp.GRB.MAXIMIZE)

    def _get_pair_callback(self, c1: int, c2: int):
        cutoff = self._get_pair_cutoff(c1, c2)

        def callback(model: gp.Model, where: int):
            if where == gp.GRB.Callback.MIPSOL:
                val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                if val < cutoff:
                    model.terminate()
        return callback

    def _get_pair_cutoff(self, c1: int, c2: int):
        mw = self.min_weight
        eps = self._eps
        return (mw * eps if c1 < c2 else 0.0)

    def _get_pair_counters(
        self,
        c1: int,
        c2: int,
        check: bool = True
    ):
        cutoff = self._get_pair_cutoff(c1, c2)
        for counter in self.get_counters(
            cutoff=cutoff,
            check=check
        ):
            yield counter
