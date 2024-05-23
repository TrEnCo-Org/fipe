from pathlib import Path
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def read_dataset(dataset: str):
    """Read dataset and return features and labels."""
    dataset_path = Path(__file__).parent / dataset
    logger.info("Loading dataset:", dataset_path)

    data = pd.read_csv(dataset_path / f'{dataset}.full.csv')

    # Read labels
    labels = data.iloc[:, -1]
    y = labels.astype('category').cat.codes
    y = y.values

    data = data.iloc[:, :-1]

    f = open(dataset_path / f'{dataset}.featurelist.csv')
    features = f.read().split(',')[:-1]
    f.close()

    return data, y, features
