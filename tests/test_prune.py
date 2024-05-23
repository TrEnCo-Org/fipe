import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
import unittest

from fipe import Features, Pruner, FullPruner
from utils import read_dataset


def get_pruner(base, X, weights):
    pruner = Pruner(base, weights)
    pruner.build()
    pruner.set_gurobi_param("OutputFlag", 0)
    pruner.add_sample_constrs(X)
    pruner.prune()
    return pruner


def get_full_pruner(base, features, X, weights):
    pruner = FullPruner(base, weights, features)
    pruner.build()
    pruner.set_gurobi_param("OutputFlag", 1)
    pruner.oracle.set_gurobi_param("OutputFlag", 0)
    pruner.add_sample_constrs(X)
    pruner.prune()
    return pruner


class TestPruner(unittest.TestCase):
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)

    # Encode features
    features = Features()
    features.fit(data)
    X = features.X.values

    # Train random forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )
    rf.fit(X_train, y_train)

    def test_prune(self):
        w = np.ones(len(self.rf))
        pruner = get_pruner(self.rf, self.X_train, w)
        pruned_weights = np.array([
            pruner.pruned_weights[t]
            for t in range(len(self.rf))
        ])
        self.assertGreaterEqual(np.sum(pruned_weights), 1)


class TestFullPruner(unittest.TestCase):
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)

    # Encode features
    features = Features()
    features.fit(data)
    X = features.X.values

    # Train random forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    rf.fit(X_train, y_train)

    # Train isolation forest
    ilf = IsolationForest(
        n_estimators=50,
        contamination=0.1
    )
    ilf.fit(X_train)

    def test_prune(self):
        w = np.ones(len(self.rf))
        pruner = get_full_pruner(
            self.rf,
            self.features,
            self.X_train,
            w
        )
        print(pruner.pruned_weights)
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
