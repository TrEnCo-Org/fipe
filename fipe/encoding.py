from copy import deepcopy

import pandas as pd

from .typing import *

class FeatureEncoder:
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
    features: dict[str, FeatureType]
    upper_bounds: dict[str, numeric]
    lower_bounds: dict[str, numeric]
    categories: dict[str, list[str]]
    inverse_categories: dict[str, str]

    columns: list[str]
    X: pd.DataFrame
    
    INF=1

    def __init__(self) -> None:
        pass

    @property
    def n_features(self):
        """
        Number of features in the data.
        """
        return len(self.features)

    @property
    def binary_features(self):
        """
        List of binary features in the data.
        """
        return [
            f for f, t in self.features.items()
            if t == FeatureType.BINARY
        ]

    @property
    def categorical_features(self):
        """
        List of categorical features in the data.
        """
        return [
            f for f, t in self.features.items()
            if t == FeatureType.CATEGORICAL
        ]

    @property
    def numerical_features(self):
        """
        List of numerical features in the data.
        """
        return [
            f for f, t in self.features.items()
            if t == FeatureType.NUMERICAL
        ]

    def fit(self, X: pd.DataFrame) -> None:
        self.data = deepcopy(X)
        
        self.clean_data()

        self.columns = self.data.columns
        self.features = dict()
        self.upper_bounds = dict()
        self.lower_bounds = dict()
        self.categories = dict()
        self.inverse_categories = dict()
        
        self.fit_binary_features()
        self.fit_categorical_features()
        self.fit_numerical_features()
        
        self.columns = self.data.columns
        self.X = self.data

    def clean_data(self) -> None:
        # Drop missing values
        self.data = self.data.dropna()
        
        # Drop columns with only one unique value
        b = self.data.nunique() > 1
        self.data = self.data.loc[:, b]
        
    def fit_binary_features(self) -> None:
        
        # For each column in the data
        # if the number of unique values is 2
        # then the feature is binary.
        # Replace the values with 0 and 1.
        for c in self.columns:
            if self.data[c].nunique() == 2:
                self.features[c] = FeatureType.BINARY
                values = self.data[c].unique()
                # TODO: do not use replace! Use map instead.
                self.data.replace({
                    c: {values[0]: 0, values[1]: 1}
                }, inplace=True)

    def fit_numerical_features(self):
        # For each column in the data
        # if the column has been identified as binary
        # then skip it. Otherwise, check if the column
        # has category dtype. If it does, skip it.
        # Otherwise, try to convert the column to numeric.
        # If the conversion is successful, then the column
        # is numerical. Convert the column to numeric
        # and store the upper and lower bounds.
        for c in self.columns:
            if c in self.features:
                continue

            if self.data[c].dtype == "category":
                continue

            x = pd.to_numeric(self.data[c], errors="coerce")
            if x.notnull().all():
                self.features[c] = FeatureType.NUMERICAL
                self.data[c] = x
                self.upper_bounds[c] = x.max() + self.INF
                self.lower_bounds[c] = x.min() - self.INF

    def fit_categorical_features(self):
        # For each column in the data
        # if the column has been identified as binary
        # or numerical, then skip it. Otherwise, the column
        # is categorical. Store the categories and
        # the inverse categories. Replace the column
        # with the encoded columns.
        
        for c in self.columns:
            if c in self.features:
                continue

            self.features[c] = FeatureType.CATEGORICAL
            df = pd.get_dummies(self.data[c], prefix=c)
            self.categories[c] = list(df.columns)
            for v in self.categories[c]:
                self.inverse_categories[v] = c
            
            # Drop the original column
            self.data.drop(columns=c, inplace=True)
            
            # Add the encoded columns
            self.data = pd.concat([self.data, df], axis=1)
            