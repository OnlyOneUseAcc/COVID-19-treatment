from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


class Splitter:

    def __init__(self, train_size=0.7, random_state=42, filter_targets=True, target_columns=None):
        self.group_splitter = GroupShuffleSplit(n_splits=2, train_size=train_size, random_state=random_state)
        self.filter_targets = filter_targets
        self.target_columns = [] if target_columns is None else target_columns

    def get_split(self, x: pd.DataFrame, group_column: Optional[str] = None):
        if group_column is not None:
            groups = x[group_column]
        else:
            groups = x.index

        if self.filter_targets and len(self.target_columns) > 0:
            values = np.unique(groups)
            non_zero = []
            groups = []
            for value in values:
                if group_column is not None:
                    temp_value = x[x[group_column] == value]
                else:
                    temp_value = x.loc[[value]]

                sum_result = np.sum(np.sum(temp_value[self.target_columns]))
                non_zero.extend([sum_result] * len(temp_value))
                if sum_result > 0:
                    groups.extend([value] * len(temp_value))
            non_zero = np.array(non_zero)

            x = x[non_zero > 0]

        train_idx, test_idx = next(self.group_splitter.split(x, groups=groups))
        return x.iloc[train_idx], x.iloc[test_idx]
