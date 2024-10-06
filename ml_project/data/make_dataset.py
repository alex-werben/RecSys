import datetime
import pandas as pd
from rectools.columns import Columns
from sklearn.calibration import LabelEncoder
import typing as tp

from rectools.model_selection.random_split import RandomSplitter
from rectools.metrics.base import MetricAtK
from rectools.metrics import Recall, Precision
from rectools.dataset import Interactions

from ml_project.common import (
    TimeRangeSplitterParams,
    LastNSplitterParams,
    RandomSplitterParams,
    InteractionsColumnParams,
    ReadParams
)

SplitterParams = tp.Union[TimeRangeSplitterParams, LastNSplitterParams, RandomSplitterParams]


def read_data(
    path: str,
    read_params: ReadParams = {}
) -> pd.DataFrame:
    # TODO: add docstring.
    return pd.read_csv(path, **read_params)

def split_data_for_train_test(
    interactions_df: pd.DataFrame,
    splitter_params: SplitterParams
) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test."""
    splitter = RandomSplitter(
        **splitter_params
    )

    interactions = Interactions(interactions_df)
    pack = splitter.split(interactions=interactions)

    for train_ids, test_ids, _ in pack:
        train_df = interactions_df.iloc[train_ids]
        test_df = interactions_df.iloc[test_ids]

    return train_df, test_df

def process_interactions(
    interactions_df: pd.DataFrame,
    interactions_column_params: InteractionsColumnParams
) -> pd.DataFrame:
    """Processes interactions dataframe."""
    le = LabelEncoder()

    interactions_df = interactions_df.rename(columns=interactions_column_params)

    interactions_df = interactions_df[interactions_df[Columns.Weight] > 0]
    interactions_df[Columns.Item] = le.fit_transform(interactions_df[Columns.Item])
    interactions_df[Columns.User] = le.fit_transform(interactions_df[Columns.User])
    interactions_df[Columns.Datetime] = datetime.date.today()

    interactions_df = (
        interactions_df
        .groupby(Columns.UserItem)
        .agg({
            Columns.Weight: "sum",
            Columns.Datetime: "last"
        })
        .reset_index()
    )

    return interactions_df

def prepare_metrics_dict(
    metric_params: tp.Dict[str, tp.Dict[str, tp.Any]]
) -> tp.Dict[str, MetricAtK]:
    metric_dict = {}

    metrics = {
        "Recall": Recall,
        "Precision": Precision
    }

    for metric_name in metric_params.keys():
        metric_dict[metric_name] = metrics[metric_name](**metric_params[metric_name])
    
    return metric_dict
