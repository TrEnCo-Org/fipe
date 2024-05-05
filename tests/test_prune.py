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

u = np.array([
True, True, False, False, True, True, False, True, True, False, False, False,
True, False, False, False, True, True, False, True, False, True, False, False,
True, True, False, True, False, False, True, True, False, True, True, False,
False, False, True, True, False, False, True, True, True, False, True, True,
False, True, False, True, False, False, False, True, False, True, True, True,
True, False, False, True, False, False, False, True, False, True, False, False,
False, False, True, False, False, False, False, False, False, True, False, True,
True, False, False, False, False, True, True, True, True, False, False, False,
False, True, True, True])

u = np.array([True, True, False, False, True, True, True, True, True, True, False, False,
False, False, False, False, True, False, False, True, False, False, False, False,
True, True, True, False, False, True, True, True, True, False, True, True,
False, False, True, True, False, False, False, True, True, False, False, True,
False, False, False, True, True, False, False, False, False, False, False, True,
True, False, False, True, False, True, False, False, True, True, False, False,
True, False, False, False, True, False, False, False, True, True, False, False,
False, False, False, False, False, True, True, True, False, True, False, False,
True, True, True, True])

from fipe import predict

y_pred = predict(rf, X_test, w)
y_pruned = predict(rf, X_test, u*w)

print("fid:", np.mean(y_pred == y_test))
exit(0)

from fipe import FIPEPrunerFull

pruner = FIPEPrunerFull(rf, w, encoder)
pruner.build()
pruner.pruner.set_gurobi_parameter('TimeLimit', 60)
pruner.pruner.set_gurobi_parameter('Threads', 47)
pruner.pruner.set_gurobi_parameter('OutputFlag', 0)
pruner.oracle.set_gurobi_parameter('OutputFlag', 0)
pruner.prune(X_train)

