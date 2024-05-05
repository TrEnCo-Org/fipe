import numpy as np
import unittest
from sklearn.ensemble import RandomForestClassifier

from fipe import FeatureEncoder, TreeEnsemble
from fipe import predict_single_proba, predict_proba, predict
from tests.utils import read_dataset


class test_predict(unittest.TestCase):
    dataset = 'Breast-Cancer-Wisconsin'

    def _get_forest_and_X(self):
        # Read dataset
        data, y, _ = read_dataset(self.dataset)

        # Encode features
        encoder = FeatureEncoder()
        encoder.fit(data)
        X = encoder.X.values

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        return rf, X

    def test_predict_single_proba(self):
        rf, X = self._get_forest_and_X()
        # Predict proba
        predict_single_proba(rf, X)

    def test_predict_proba(self):
        rf, X = self._get_forest_and_X()

        # - Test 1: w=1/n -
        nb_trees = len(rf)
        weights = np.ones(nb_trees) * 1/nb_trees
        # Predict proba
        predict_proba(rf, X, weights)

        # - Test 2: w uniform -
        weights = np.random.uniform(size=(nb_trees))
        weights = weights / np.sum(weights)
        # Predict proba
        predict_proba(rf, X, weights)

    def test_predict(self):
        rf, X = self._get_forest_and_X()

        # - Test 1: w=1/n -
        nb_trees = len(rf)
        weights = np.ones(nb_trees) * 1/nb_trees
        # Predict proba
        predict(rf, X, weights)

        # - Test 2: w uniform -
        weights = np.random.uniform(size=(nb_trees))
        weights = weights / np.sum(weights)
        # Predict proba
        predict(rf, X, weights)
