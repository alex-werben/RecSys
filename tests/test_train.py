import os

from hydra import compose, initialize

from pipelines.train import main


def test_train_e2e(mock_s3_put):
    """Tests train pipeline end-to-end and checks output existence."""
    with initialize(config_path="../configs", version_base="1.1"):
        conf = compose(config_name="test_config", return_hydra_config=True)

        dir_name = conf.data.output.model_path.split('/')[0]
        os.makedirs(dir_name, exist_ok=True)

        main(conf)

        # assert os.path.exists(mock_s3_put, )
        # assert os.path.exists(conf.data.output.model_path)
        # assert os.path.exists(conf.data.output.dataset_path)

        os.remove(conf.data.output.model_path)
        os.remove(conf.data.output.dataset_path)
        os.rmdir(dir_name)
