import logging
from pathlib import Path
import sys

import hydra
import mlflow
from omegaconf import DictConfig
from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.metrics import calc_metrics
from sklearn.preprocessing import LabelEncoder

PROJECT_PATH = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_PATH)

from ml_project.data.make_dataset import prepare_metrics_dict
from ml_project.models.model_fit_predict import train_model

from ml_project.data import (
    read_data,
    split_data_for_train_test
)
from dvc.api import get_url

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(
    version_base="1.1",
    config_path="../configs",
    config_name="train_config"
)
def main(conf: DictConfig):
    with mlflow.start_run() as run:
        data_url = get_url(
            path=conf.data.output.interactions_path,
            repo=PROJECT_PATH,
            remote="local_storage",
            rev=conf.version
        )

        logger.info(f"{data_url=}")

        interactions_df = read_data(
            path=data_url,
            read_params=conf.data.input.interactions.read_params
        )
        le = LabelEncoder()
        interactions_df[Columns.Item] = le.fit_transform(interactions_df[Columns.Item]) # TODO: move to preprocessing
        interactions_df.info()
        
        mlflow.log_param("interactions_version", conf.version)
        mlflow.log_param("interactions_shape", interactions_df.shape)
        
        logger.info(f"Interactions version: {conf.version}")
        logger.info(f"{interactions_df.shape=}")

        mlflow.log_artifact(
            local_path=conf.data.output.interactions_path,
            artifact_path="datasets"
        )

        logger.info("interactions_df.info():")
        interactions_df.info()

        train_df, test_df = split_data_for_train_test(
            interactions_df=interactions_df,
            splitter_params=conf.splitter_params
        )

        logger.info(f"{train_df.shape=}")
        logger.info(f"{test_df.shape=}")

        test_df = test_df[test_df[Columns.User].isin(train_df[Columns.User])]

        logger.info(f"{test_df.shape=}")

        dataset = Dataset.construct(
            interactions_df=interactions_df
        )

        model = train_model(
            dataset=dataset,
            train_params=conf.train_params
        )

        users_to_predict = test_df[Columns.User].unique()

        recs_df = model.recommend(
            users=users_to_predict,
            dataset=dataset,
            **conf.predict_params
        )

        metrics_dict = prepare_metrics_dict(conf.metric_params)

        metrics = calc_metrics(
            metrics=metrics_dict,
            reco=recs_df,
            interactions=test_df,
            prev_interactions=train_df,
        )

        mlflow.log_metrics(metrics=metrics)

if __name__ == "__main__":
    main()
