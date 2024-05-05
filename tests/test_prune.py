import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
import unittest

from fipe import FeatureEncoder, FIPEPruner, FIPEPrunerFull
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


class TestFullPruner(unittest.TestCase):
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)

    # Encode features
    encoder = FeatureEncoder()
    encoder.fit(data)
    X = encoder.X.values

    # Train random forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42)
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X_train, y_train)

    # Train isolation forest
    ilf = IsolationForest(n_estimators=50, contamination=0.1)
    ilf.fit(X_train)

    def test_prune(self):
        w = np.ones(len(self.rf))
        full_pruner = FIPEPrunerFull(self.rf, w,
                                     self.ilf, self.encoder,
                                     max_iter=3)
        full_pruner.build()
        full_pruner.add_points(self.X_test)
        full_pruner.prune()

        # Read solution
        active_trees = full_pruner.pruner.active
        self.assertEqual(len(active_trees), len(self.rf))
        self.assertTrue(set([0, 1]).issuperset(set(active_trees)))
        self.assertGreaterEqual(np.sum(active_trees), 1)
        self.assertLessEqual(np.sum(active_trees), len(self.rf)-1)
