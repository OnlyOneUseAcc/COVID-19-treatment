from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import pandas as pd
import numpy as np


class CategoricalPreprocessor(FunctionTransformer):

    def __init__(self, excluded_columns=None, categorical_threshold=20):
        super().__init__()
        self.final_column_names = []
        self.excluded_columns = set() if excluded_columns is None else set(excluded_columns)
        self.categorical_threshold = categorical_threshold
        self.encoder = OneHotEncoder(sparse=False)
        self.encoded_columns = []

    def fit(self, x: pd.DataFrame, y=None):
        for column in x.columns:
            if column in self.excluded_columns or x[column].value_counts().shape[0] == 2:
                continue
            if x[column].value_counts().shape[0] < self.categorical_threshold:
                self.encoded_columns.append(column)

        self.encoder.fit(x[self.encoded_columns])
        return self

    def transform(self, x: pd.DataFrame):
        encoded_values = self.encoder.transform(x[self.encoded_columns])
        x = x[list(set(x.columns).difference(set(self.encoded_columns)))]

        x = pd.concat(
            [
                x,
                pd.DataFrame(
                    data=encoded_values,
                    columns=self.encoder.get_feature_names_out(),
                    index=x.index
                ),
             ],
            axis=1
        )

        self.final_column_names = x.columns
        return x

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.final_column_names)
