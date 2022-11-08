from typing import Tuple, List

import joblib
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from preprocessing import CategoricalPreprocessor
from preprocessing import DinamChangesInserter


class XGBPipeline:

    def __init__(
            self,
            dinam_fact_columns: List[str]
    ):
        preprocessors = Pipeline(
            [
                ('encoder', CategoricalPreprocessor()),
                ('change_inserter', DinamChangesInserter(use_columns=dinam_fact_columns)),
                ('imputer', KNNImputer(n_neighbors=20)),
                ('scaler', MinMaxScaler())

            ]
        )

        self.pipe = Pipeline([
            ('preprocessors', preprocessors),
            ('xgb', xgb.XGBClassifier(
                max_depth=5,
                n_estimators=10000,
                learning_rate=0.001,
                subsample=0.5,
                objective="binary:logistic",
                random_state=42,
                verbosity=0,
                early_stopping_rounds=30
                )
            )
        ])

    def fit(self, x, y, val_data: Tuple[pd.DataFrame, pd.DataFrame] = None, **fit_params):
        x_train_transformed = self.pipe[0].fit_transform(x)

        if val_data is not None:
            x_val_transformed = self.pipe[0].transform(val_data[0])
            fit_params.update({'eval_set': [(x_val_transformed, val_data[1])]})

        self.pipe[-1].fit(
            x_train_transformed,
            y,
            **fit_params
        )

    def predict(self, x):
        return self.pipe.predict(x)

    @staticmethod
    def load(path: str):
        return joblib.load(path)

    def save(self, path: str):
        joblib.dump(self, path)
