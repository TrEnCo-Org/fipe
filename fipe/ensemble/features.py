from copy import deepcopy

import pandas as pd

from ..typing import (
    FeatureType,
    numeric
)


class Features:
    """
    Encoder class for encoding data

    Attributes:
    -----------
    features: dict[str, FeatureType]
        Dictionary of features and their types.

    upper_bounds: dict[str, numeric]
        Dictionary of upper bounds for numerical features.

    lower_bounds: dict[str, numeric]
        Dictionary of lower bounds for numerical features.

    categories: dict[str, list[str]]
        Dictionary of categories for categorical features.

    inverse_categories: dict[str, str]
        Dictionary of inverse categories for categorical features.

    columns: list[str]
        List of columns in the data after encoding.

    X: pd.DataFrame
        Dataframe of the data after encoding.
    """
    types: dict[str, FeatureType]
    upper_bounds: dict[str, numeric]
    lower_bounds: dict[str, numeric]
    values: dict[str, list[numeric]]
    categories: dict[str, list[str]]
    inverse_categories: dict[str, str]

    columns: list[str]
    X: pd.DataFrame

    INF = 1

    def __init__(self) -> None:
        pass

    @property
    def n_features(self):
        """
        Number of features in the data.
        """
        return len(self.types)

    @property
    def binary(self):
        """
        List of binary features in the data.
        """
        def fn(f):
            return self.types[f] == FeatureType.BINARY
        return set(filter(fn, self.types.keys()))

    @property
    def categorical(self):
        """
        List of categorical features in the data.
        """
        def fn(f):
            return self.types[f] == FeatureType.CATEGORICAL
        return set(filter(fn, self.types.keys()))

    @property
    def discrete(self):
        """
        List of discrete features in the data.
        """
        def fn(f):
            return self.types[f] == FeatureType.DISCRETE
        return set(filter(fn, self.types.keys()))

    @property
    def continuous(self):
        """
        List of continuous features in the data.
        """
        def fn(f):
            return self.types[f] == FeatureType.CONTINUOUS
        return set(filter(fn, self.types.keys()))

    @property
    def numerical(self):
        """
        List of numerical features in the data.
        """
        return self.continuous | self.discrete

    def fit(
        self,
        X: pd.DataFrame,
        discrete: set[str] | None = None,
    ) -> None:
        self.data = deepcopy(X)

        self._clean_data()

        self.columns = list(self.data.columns)
        self.types = dict()
        self.upper_bounds = dict()
        self.lower_bounds = dict()
        self.values = dict()
        self.categories = dict()
        self.inverse_categories = dict()

        self._fit_binary_features()
        self._fit_discrete_features(discrete)
        self._fit_continuous_features()
        self._fit_categorical_features()
        self._save_columns()

        self.X = self.data

    def _clean_data(self) -> None:
        # Drop missing values
        self.data = self.data.dropna()
        # Drop columns with only one unique value
        b = self.data.nunique() > 1
        self.data = self.data.loc[:, b]

    def _fit_binary_features(self):
        # For each column in the data
        # if the number of unique values is 2
        # then the feature is binary.
        # Replace the values with 0 and 1.
        for c in self.columns:
            if self.data[c].nunique() == 2:
                self.types[c] = FeatureType.BINARY
                df = pd.get_dummies(self.data[c], drop_first=True)
                self.data.drop(columns=c, inplace=True)
                self.data[c] = df.iloc[:, 0]

    def _fit_discrete_features(
        self,
        discrete: set[str] | None = None
    ):
        for c in self.columns:
            if c in self.types:
                continue
            if discrete is not None and c in discrete:
                x = pd.to_numeric(self.data[c], errors="coerce")
                if x.notnull().all():
                    self.types[c] = FeatureType.DISCRETE
                    self.data[c] = x
                    values = x.unique()
                    values.sort()
                    self.values[c] = list(values)

    def _fit_continuous_features(self):
        # For each column in the data
        # if the column has been identified as binary
        # then skip it. Otherwise, check if the column
        # has category dtype. If it does, skip it.
        # Otherwise, try to convert the column to numeric.
        # If the conversion is successful, then the column
        # is numerical. Convert the column to numeric
        # and store the upper and lower bounds.
        for c in self.columns:
            if c in self.types:
                continue

            if self.data[c].dtype == "category":
                continue

            x = pd.to_numeric(self.data[c], errors="coerce")
            if x.notnull().all():
                self.types[c] = FeatureType.CONTINUOUS
                self.data[c] = x
                self.upper_bounds[c] = x.max() + self.INF
                self.lower_bounds[c] = x.min() - self.INF

    def _fit_categorical_features(self):
        # For each column in the data
        # if the column has been identified as binary
        # or numerical, then skip it. Otherwise, the column
        # is categorical. Store the categories and
        # the inverse categories. Replace the column
        # with the encoded columns.
        for c in self.columns:
            if c in self.types:
                continue

            self.types[c] = FeatureType.CATEGORICAL
            df = pd.get_dummies(self.data[c], prefix=c)
            self.categories[c] = list(df.columns)
            for v in self.categories[c]:
                self.inverse_categories[v] = c
            # Drop the original column
            self.data.drop(columns=c, inplace=True)
            # Add the encoded columns
            self.data = pd.concat([self.data, df], axis=1)

    def _save_columns(self):
        self.columns = list(self.data.columns)
