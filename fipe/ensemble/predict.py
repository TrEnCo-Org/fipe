import numpy as np

from sklearn.ensemble._base import BaseEnsemble

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def predict_proba_base(e, X):
    """
    Predict the class probabilities of each sample in X
    according to the estimator e.
    """
    if X.ndim == 1:
        # If X is a single sample,
        # reshape it to a 2D array
        # with a single row.
        X = X.reshape(1, -1)

    p = e.predict_proba(X)
    k = p.shape[-1]
    q = np.argmax(p, axis=-1)
    p = np.eye(k)[q]

    return p


def predict_single_proba(E: BaseEnsemble, X):
    """
    Predict the class probabilities of each sample in X
    according to each estimator in the ensemble E.
    """
    if X.ndim == 1:
        # If X is a single sample,
        # reshape it to a 2D array
        # with a single row.
        X = X.reshape(1, -1)

    def fn(e):
        return predict_proba_base(e, X)
    p = np.stack(list(map(fn, E)))

    # The shape of p is (n_estimators, n_samples, n_classes).
    # We want to swap the axes to have the shape
    # (n_samples, n_estimators, n_classes).
    p = np.swapaxes(p, 0, 1)
    return p


def predict_proba(E: BaseEnsemble, X, w):
    """
    Predict the score of each sample in X
    according to the ensemble E.
    """
    w = np.array(w)
    p = predict_single_proba(E, X)
    return np.average(p, axis=1, weights=w)


def predict(E: BaseEnsemble, X, w):
    """
    Predict the class of each sample in X using
    the model ensemble E and the weights w.
    """
    p = predict_proba(E, X, w)
    return np.argmax(p, axis=-1)
