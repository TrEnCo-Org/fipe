from sklearn.ensemble import IsolationForest

from .features import Features
from .ensemble import Ensemble


class IsolationEnsemble(Ensemble):
    max_samples: int
    offset: float

    def __init__(
        self,
        isolation_forest: IsolationForest,
        features: Features
    ):
        Ensemble.__init__(self, isolation_forest, features)
        self.max_samples = isolation_forest.max_samples_
        self.offset = isolation_forest.offset_
