import datetime
import pandas as pd
from rectools import Columns
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from ml_project.common import InteractionsColumnParams


class InteractionsTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for interactions."""
    def __init__(
        self,
        interactions_column_params: InteractionsColumnParams,
    ) -> None:
        """Initialize transformer.

        Args:
            interactions_column_params (InteractionsColumnParams): column mapper parameters
        """
        column_name_mapper = interactions_column_params.items()
        inverse_column_name_mapper = {v: k for k, v in column_name_mapper}
        self.interactions_column_params = inverse_column_name_mapper
        self._encoder = LabelEncoder()

    def fit(
        self,
        X: pd.DataFrame
    ):
        """Fit transformer

        Args:
            X (pd.DataFrame): interactions

        Returns:
            _type_: self
        """
        self.X = X

        return self

    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Transform interactions.

        Args:
            X (pd.DataFrame): interactions

        Returns:
            pd.DataFrame: transformed interactions
        """
        X = X.rename(columns=self.interactions_column_params)
        X = X[X[Columns.Weight] > 0]
        X[Columns.User] = self._encoder.fit_transform(X[Columns.User])
        X[Columns.Item] = self._encoder.fit_transform(X[Columns.Item])
        X[Columns.Datetime] = datetime.date.today()

        X = (
            X
            .groupby(Columns.UserItem)
            .agg({
                Columns.Weight: "sum",
                Columns.Datetime: "last"
            })
            .reset_index()
        )

        return X
