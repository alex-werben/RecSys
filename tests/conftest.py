import os
import pytest
from pathlib import Path

@pytest.fixture()
def dataset_path():
    directory = str(Path(__file__).parent)
    path = os.path.join(directory, "dataset_example.csv")
    
    return path
