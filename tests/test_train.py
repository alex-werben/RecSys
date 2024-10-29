import os
import shutil

from hydra import compose, initialize

from pipelines import train


def test_train_e2e(mock_s3_put):
    """Tests train pipeline end-to-end and checks output existence."""
    with initialize(config_path="../configs", version_base="1.1"):
        conf = compose(config_name="train_svd", return_hydra_config=True)

        dir_name = conf.data.output.model_path.split('/')[0]
        os.makedirs(dir_name, exist_ok=True)

        conf.data.input.interactions.path.processed = "tests/dataset_example.csv"

        train(conf)

        assert os.path.exists(conf.data.output.model_path)
        assert os.path.exists(conf.data.output.dataset_path)

        shutil.rmtree(dir_name, ignore_errors=True)
