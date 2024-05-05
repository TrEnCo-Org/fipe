import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
import unittest

from fipe import FeatureEncoder, FIPEOracle, TreeEnsemble
from tests.utils import read_dataset


class TestOracle(unittest.TestCase):
    # Setup test
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)

    # Encode features
    encoder = FeatureEncoder()
    encoder.fit(data)
    X = encoder.X.values

    # Train random forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Train isolation forest
    ilf = IsolationForest(n_estimators=50, contamination=0.1)
    ilf.fit(X_train)

    def test_oracle_fails_with_all_active(self):
        weights = np.ones(len(self.rf))
        tree_ensemble = TreeEnsemble(self.rf, self.encoder)
        isolation_ensemble = TreeEnsemble(self.ilf, self.encoder)
        # Build oracle
        oracle = FIPEOracle(self.encoder,
                            tree_ensemble,
                            isolation_ensemble,
                            weights,
                            self.ilf.max_samples_,
                            self.ilf.offset_)
        oracle.build()
        # Separate with all trees selected
        active_trees = np.ones(len(self.rf))
        X = oracle.separate(active_trees)
        self.assertTrue(len(X) == 0)

    def test_oracle_succeeds_with_single_tree(self):
        weights = np.ones(len(self.rf))
        tree_ensemble = TreeEnsemble(self.rf, self.encoder)
        isolation_ensemble = TreeEnsemble(self.ilf, self.encoder)
        # Build oracle
        oracle = FIPEOracle(self.encoder,
                            tree_ensemble,
                            isolation_ensemble,
                            weights,
                            self.ilf.max_samples_,
                            self.ilf.offset_)
        oracle.build()
        # Separate with all trees selected
        active_trees = np.zeros(len(self.rf))
        active_trees[0] = 1
        X = oracle.separate(active_trees)
        self.assertTrue(len(X) > 0)
