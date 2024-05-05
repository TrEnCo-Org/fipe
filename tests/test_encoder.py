import unittest

from fipe import FeatureEncoder
from tests.utils import read_dataset


class TestFeatureEncoder(unittest.TestCase):
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)
    true_features = ['Clump-T', 'Uniformity-Size', 'Uniformity-Shape',
                     'Adhesion', 'Cell-Size', 'Bare-Nuclei', 'Bland-Chromatin',
                     'Normal-Nucleoli', 'Mitoses']
    true_value_set = set([float(i) for i in list(range(1, 11))])

    def test_encoder(self):
        encoder = FeatureEncoder()
        # - Test 1: continuous features -
        encoder.fit(self.data)
        self.assertEqual(encoder.n_features, 9)
        self.assertEqual(encoder.binary_features, [])
        self.assertEqual(encoder.categorical_features, [])
        self.assertEqual(encoder.continuous_features, self.true_features)
        self.assertEqual(encoder.discrete_features, [])
        self.assertEqual(encoder.numerical_features, self.true_features)

        # - Test 2: discrete features -
        encoder.fit(self.data, discrete_features=self.true_features)
        self.assertEqual(encoder.n_features, 9)
        self.assertEqual(encoder.binary_features, [])
        self.assertEqual(encoder.categorical_features, [])
        self.assertEqual(encoder.continuous_features, [])
        self.assertEqual(encoder.discrete_features, self.true_features)
        self.assertEqual(encoder.numerical_features, self.true_features)

        # Check all discrete features have 10 values from 0. to 9.
        for f in encoder.discrete_features:
            value_set = set(encoder.values[f])
            self.assertTrue(self.true_value_set.issuperset(value_set))
