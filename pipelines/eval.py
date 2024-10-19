import logging
from pathlib import Path
import sys

import hydra
import mlflow
from omegaconf import DictConfig
import pandas as pd
from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.metrics import calc_metrics

PROJECT_PATH = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_PATH)

from ml_project.models import train_model
from ml_project.data import (
    prepare_metrics_dict,
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
    config_name="train_svd"
)
def evaluate(conf: DictConfig):
    with mlflow.start_run():
        data_url = get_url(
            path=conf.data.input.interactions.path.processed,
            repo=PROJECT_PATH,
            remote="local_storage",
            rev=conf.interactions_version
        )

        logger.info(f"{data_url=}")

        interactions_df = pd.read_csv(data_url)

        interactions_df.info()

        mlflow.log_param("interactions_version", conf.interactions_version)
        mlflow.log_param("interactions_shape", interactions_df.shape)

        logger.info(f"Interactions version: {conf.interactions_version}")
        logger.info(f"{interactions_df.shape=}")

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
        mlflow.log_param("train_df_shape", train_df.shape)
        mlflow.log_param("test_df_shape", test_df.shape)

        dataset = Dataset.construct(
            interactions_df=interactions_df
        )

        mlflow.log_param("model_type", conf.train_params.model_type)
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
    evaluate()
