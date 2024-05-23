import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
import unittest

from fipe import Features, Oracle, Ensemble
from utils import read_dataset


class TestOracle(unittest.TestCase):
    # Setup test
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)

    # Encode features
    features = Features()
    features.fit(data)
    X = features.X.values

    # Train random forest
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=5,
        random_state=42,
        max_depth=2
    )
    rf.fit(X_train, y_train)
    ensemble = Ensemble(rf, features)
    weights = np.ones(len(rf))

    # Train isolation forest
    ilf = IsolationForest(
        n_estimators=50,
        contamination=0.1,
        random_state=42
    )
    ilf.fit(X_train)

    def test_oracle_fails_with_all_active(self):
        oracle = Oracle(
            self.features,
            self.ensemble,
            self.weights
        )
        oracle.build()
        oracle.set_gurobi_param("OutputFlag", 0)
        # Separate with all trees selected
        active_trees = {
            t: 1.0 for t in range(len(self.rf))
        }
        X = list(oracle.separate(active_trees))
        self.assertTrue(len(X) == 0)

    def test_oracle_succeeds_with_single_tree(self):
        oracle = Oracle(
            self.features,
            self.ensemble,
            self.weights
        )
        oracle.build()
        oracle.set_gurobi_param("OutputFlag", 0)
        # Separate with all trees selected
        active_trees = {
            t: 1.0 for t in range(len(self.rf))
        }
        active_trees[0] = 0
        X = list(oracle.separate(active_trees))
        self.assertTrue(len(X) > 0)


if __name__ == '__main__':
    unittest.main()
