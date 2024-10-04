import datetime
import pandas as pd
from rectools.columns import Columns
from sklearn.calibration import LabelEncoder
import typing as tp

from sklearn.model_selection import train_test_split
from rectools.model_selection.random_split import RandomSplitter
from rectools.dataset import Interactions

from ml_project.common import TimeRangeSplitterParams, LastNSplitterParams, RandomSplitterParams

SplitterParams = tp.Union[TimeRangeSplitterParams, LastNSplitterParams, RandomSplitterParams]


def read_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="unicode-escape") # TODO: add params to config

def merge_interactions_and_items(
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame
) -> pd.DataFrame:
    """Merges interactions with item features."""
    merged_df = (
        interactions_df
        .merge(
            items_df[["ISBN", "Book-Title", "Image-URL-M"]],
            how="left",
            on="ISBN"
        )
    )

    return merged_df

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
    interactions_df: pd.DataFrame
) -> pd.DataFrame:
    """Processes interactions dataframe."""
    le = LabelEncoder()

    interactions_df = interactions_df.rename(
        columns={
            "User-ID": Columns.User,
            "ISBN": Columns.Item,
            "Book-Rating": Columns.Weight,
            "Book-Title": "item_name",
            "Image-URL-M": "image_url"
        }
    )

    # interactions_df = interactions_df[interactions_df["item_name"].notna()]
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

