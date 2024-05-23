import unittest

from fipe import Features
from utils import read_dataset


class TestFeatures(unittest.TestCase):
    dataset = 'Breast-Cancer-Wisconsin'
    data, y, _ = read_dataset(dataset)
    true_features = set([
        'Clump-T',
        'Uniformity-Size',
        'Uniformity-Shape',
        'Adhesion',
        'Cell-Size',
        'Bare-Nuclei',
        'Bland-Chromatin',
        'Normal-Nucleoli',
        'Mitoses'
    ])
    true_value_set = set([float(i) for i in list(range(1, 11))])

    def test_encoder(self):
        features = Features()
        # - Test 1: continuous features -
        features.fit(self.data)
        self.assertEqual(features.n_features, 9)
        self.assertEqual(len(features.binary), 0)
        self.assertEqual(len(features.categorical), 0)
        self.assertEqual(features.continuous, self.true_features)
        self.assertEqual(len(features.discrete), 0)
        self.assertEqual(features.numerical, self.true_features)

        # - Test 2: discrete features -
        features.fit(self.data, discrete=set(self.true_features))
        self.assertEqual(features.n_features, 9)
        self.assertEqual(len(features.binary), 0)
        self.assertEqual(len(features.categorical), 0)
        self.assertEqual(len(features.continuous), 0)
        self.assertEqual(features.discrete, self.true_features)
        self.assertEqual(features.numerical, self.true_features)

        # Check all discrete features have 10 values from 0. to 9.
        for f in features.discrete:
            value_set = set(features.values[f])
            self.assertTrue(self.true_value_set.issuperset(value_set))


if __name__ == '__main__':
    unittest.main()
