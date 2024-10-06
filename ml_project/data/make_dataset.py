import datetime
import pandas as pd
from rectools.columns import Columns
from sklearn.calibration import LabelEncoder
import typing as tp

from rectools.model_selection.random_split import RandomSplitter
from rectools.metrics.base import MetricAtK
from rectools.metrics import Precision, Recall
from rectools.dataset import Interactions

from ml_project.common import (
    InteractionsColumnParams,
    LastNSplitterParams,
    RandomSplitterParams,
    ReadParams,
    TimeRangeSplitterParams
)

SplitterParams = tp.Union[TimeRangeSplitterParams, LastNSplitterParams, RandomSplitterParams]


def read_data(
    path: str,
    read_params: ReadParams = {}
) -> pd.DataFrame:
    """Read data from .csv file.

    Args:
        path (str): path to .csv
        read_params (ReadParams, optional): read_csv method parameters. Defaults to {}.

    Returns:
        pd.DataFrame: DataFrame with data
    """
    return pd.read_csv(path, **read_params)


def split_data_for_train_test(
    interactions_df: pd.DataFrame,
    splitter_params: SplitterParams
) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test.

    Args:
        interactions_df (pd.DataFrame): dataframe with interactions
        splitter_params (SplitterParams): splitter paramters

    Returns:
        tp.Tuple[pd.DataFrame, pd.DataFrame]: train and test interactions
    """
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
    """Process interactions.

    Args:
        interactions_df (pd.DataFrame): dataframe with interactions
        interactions_column_params (InteractionsColumnParams): interactions column parameters

    Returns:
        pd.DataFrame: processed interactions
    """
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
    """Prepare dict with metric names as keys and metric classes as values.

    Args:
        metric_params (tp.Dict[str, tp.Dict[str, tp.Any]]): metric parameters

    Returns:
        tp.Dict[str, MetricAtK]: dict with metric names and classes
    """
    metric_dict = {}

    metrics = {
        "Recall": Recall,
        "Precision": Precision
    }

    for metric_name in metric_params.keys():
        metric_dict[metric_name] = (
            metrics[metric_name](**metric_params[metric_name])
        )

    return metric_dict
