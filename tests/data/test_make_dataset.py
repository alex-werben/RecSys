import sys
from pathlib import Path
print(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_project.data import read_data, split_data_for_train_test, process_interactions
from ml_project.common import ReadParams, RandomSplitterParams, InteractionsColumnParams

def test_load_dataset(
    dataset_path: str,
) -> None:
    interactions_df = read_data(dataset_path)
    
    assert interactions_df.shape[1] == 4

def test_split_data_from_train_test(
    dataset_path: str
) -> None:
    interactions_df = read_data(dataset_path)
    splitter_params = RandomSplitterParams(test_fold_frac=0.2)

    train_df, test_df = split_data_for_train_test(
        interactions_df=interactions_df,
        splitter_params=splitter_params.__dict__
    )
    
    assert train_df.shape[0] > test_df.shape[0]
    
def test_process_interactions(
    dataset_path: str
) -> None:
    interactions_df = read_data(dataset_path)
    assert interactions_df.shape[0] == 100
    
    interactions_column_params = {v: k for k, v in InteractionsColumnParams().__dict__.items()}

    interactions_df = process_interactions(
        interactions_df=interactions_df,
        interactions_column_params=interactions_column_params
    )
    
    assert interactions_df.shape[0] == 85