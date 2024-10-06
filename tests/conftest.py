import os
from pathlib import Path

import pytest


@pytest.fixture()
def dataset_path():
    """Path to synthetic dataset."""
    directory = str(Path(__file__).parent)
    path = os.path.join(directory, "dataset_example.csv")

    return path
