import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import unittest

from fipe import FeatureEncoder, FIPEPruner
from tests.utils import read_dataset


class TestPruner(unittest.TestCase):
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)

    # Encode features
    encoder = FeatureEncoder()
    encoder.fit(data)
    X = encoder.X.values

    # Train random forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    def test_prune(self):
        w = np.ones(len(self.rf))
        pruner = FIPEPruner(self.rf, w)
        pruner.build()
        pruner.add_constraints(self.X_test)
        pruner.prune()
        # Read solution
        active_trees = pruner.active
        self.assertEqual(len(active_trees), len(self.rf))
        self.assertTrue(set([0, 1]).issuperset(set(active_trees)))
        self.assertGreaterEqual(np.sum(active_trees), 1)
