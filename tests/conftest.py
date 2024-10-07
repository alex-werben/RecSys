import os
from pathlib import Path
import sys
from unittest import mock
import tempfile

import pytest

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.models import serialize_object


@pytest.fixture()
def dataset_path():
    """Path to synthetic dataset."""
    directory = str(Path(__file__).parent)
    path = os.path.join(directory, "dataset_example.csv")

    return path


@pytest.fixture
def mock_s3_put():
    with mock.patch("pipelines.train.S3Connector.put") as mocked_put:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mocked_put.side_effect = lambda obj, path: serialize_object(obj, path)
            yield tmp_dir
