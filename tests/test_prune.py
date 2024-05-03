from pathlib import Path
import pandas as pd

dataset = 'Breast-Cancer-Wisconsin'
dataset_path = Path(__file__).parent / dataset

print(dataset_path)

data = pd.read_csv(dataset_path / f'{dataset}.full.csv')
target = data.iloc[:, -1]
data = data.iloc[:, :-1]

f = open(dataset_path / f'{dataset}.featurelist.csv')
features = f.read().split(',')[:-1]
f.close()

from fipe import FeatureEncoder

encoder = FeatureEncoder()
encoder.fit(data)

X = encoder.X.values
y = target.astype('category').cat.codes
y = y.values

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

import numpy as np
w = np.ones(len(rf))

from fipe import FIPEPrunerFull

pruner = FIPEPrunerFull(rf, w, encoder)
pruner.build()
pruner.prune(X_train)

