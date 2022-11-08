from typing import Optional, List

import pandas as pd
from sklearn.preprocessing import FunctionTransformer
import numpy as np


class DinamChangesInserter(FunctionTransformer):

    def __init__(
            self,
            empty_value=0,
            group_column: Optional[str] = None,
            sort_column: Optional[str] = None,
            use_columns: Optional[List[str]] = None
    ):
        super().__init__()
        self.final_columns = []
        self.empty_value = empty_value
        self.group_column = group_column
        self.sort_column = sort_column
        self.use_columns = use_columns

    def fit(self, X, y=None):
        self.final_columns = list(X.columns)
        self.use_columns = list(set(self.use_columns).intersection(set(X.columns)))
        self.final_columns.extend([f'{column}_change' for column in self.use_columns])
        return self

    def __index_iter(self, X: pd.DataFrame, indexes):
        for index in indexes:
            values = X.loc[[index]]
            yield values.sort_values(by=self.sort_column) if self.sort_column is not None else values

    def __column_iter(self, X: pd.DataFrame, column_values):
        for column_value in column_values:
            values = X[X[self.group_column] == column_value]
            yield values.sort_values(by=self.sort_column) if self.sort_column is not None else values

    def transform(self, X: pd.DataFrame):
        values = []
        if self.use_columns is None:
            self.use_columns = X.columns

        if self.group_column is None:
            values_iter = self.__index_iter(X, pd.unique(X.index))
        else:
            values_iter = self.__column_iter(X, pd.unique(X[self.group_column]))

        for group in values_iter:
            if self.use_columns is not None:
                group = group[self.use_columns]

            change_values = np.empty(shape=group.shape)
            change_values[1:, :] = group.values[:-1, :] - group.values[1:, :]
            change_values[0, :] = [self.empty_value] * group.values.shape[1]

            values.append(change_values)
        values = np.concatenate(values, axis=0)
        X = pd.concat([
            X,
            pd.DataFrame(
                data=values,
                columns=[f'{column}_change' for column in self.use_columns],
                index=X.index
            )
        ],
            axis=1
        )
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.final_columns)
