import numpy as np
import unittest
from sklearn.ensemble import RandomForestClassifier

from fipe import FeatureEncoder
from fipe import predict_single_proba, predict_proba, predict
from tests.utils import read_dataset


class test_predict(unittest.TestCase):
    # Setup test
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)
    # Encode features
    encoder = FeatureEncoder()
    encoder.fit(data)
    X = encoder.X.values
    x_0 = X[0]
    # Train random forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    def test_predict_single_proba(self):
        scores = predict_single_proba(self.rf, self.X)
        self.assertEqual(len(scores), len(self.X))
        self.assertEqual(len(scores[0]), len(self.rf))
        self.assertEqual(len(scores[0][0]), 2)
        score = predict_single_proba(self.rf, self.x_0)
        self.assertEqual(len(score), 1)
        self.assertEqual(len(score[0]), len(self.rf))
        self.assertTrue(set([0, 1]).issuperset(set(score[0][0])))

    def test_predict_proba(self):
        # - Test 1: w=1/n -
        nb_trees = len(self.rf)
        weights = np.ones(nb_trees) * 1/nb_trees
        # Predict proba
        scores = predict_proba(self.rf, self.X, weights)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue((0.0 <= scores).all())
        self.assertTrue((len(self.rf) > scores).all())

        # - Test 2: w uniform -
        weights = np.random.uniform(size=(nb_trees))
        weights = weights / np.sum(weights)
        # Predict proba
        scores = predict_proba(self.rf, self.X, weights)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue((0.0 <= scores).all())
        self.assertTrue((len(self.rf) > scores).all())

        # - Test 3: single input -
        score = predict_proba(self.rf, self.x_0, weights)
        self.assertEqual(len(score), 1)
        self.assertEqual(len(score[0]), 2)
        self.assertTrue((0.0 <= score).all())
        self.assertTrue((len(self.rf) > score).all())

    def test_predict(self):
        # - Test 1: w=1/n -
        nb_trees = len(self.rf)
        weights = np.ones(nb_trees) * 1/nb_trees
        # Predict proba
        classes = predict(self.rf, self.X, weights)
        self.assertEqual(len(classes), len(self.X))
        self.assertTrue(set([0, 1]).issuperset(set(classes)))

        # - Test 2: w uniform -
        weights = np.random.uniform(size=(nb_trees))
        weights = weights / np.sum(weights)
        # Predict proba
        classes = predict(self.rf, self.X, weights)
        self.assertEqual(len(classes), len(self.X))
        self.assertTrue(set([0, 1]).issuperset(set(classes)))

        # - Test 3: single input -
        classs = predict(self.rf, self.x_0, weights)
        self.assertEqual(len(classs), 1)
        self.assertTrue(classs in [0, 1])
        self.assertTrue(classs[0] in [0, 1])
