import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_project.common import RandomSplitterParams
from ml_project.data import (
    read_data,
    split_data_for_train_test
)


def test_load_dataset(
    dataset_path: str,
) -> None:
    """Tests reading dataset and correctness of interactions shape.

    Args:
        dataset_path (str): path to dataset
    """
    interactions_df = read_data(dataset_path)

    assert interactions_df.shape[1] == 4


def test_split_data_from_train_test(
    dataset_path: str
) -> None:
    """Tests reading dataset and correctness of splitting into train and test.

    Args:
        dataset_path (str): path to dataset
    """
    interactions_df = read_data(dataset_path)
    splitter_params = RandomSplitterParams(test_fold_frac=0.2)

    train_df, test_df = split_data_for_train_test(
        interactions_df=interactions_df,
        splitter_params=splitter_params.__dict__
    )

    assert train_df.shape[0] > test_df.shape[0]
