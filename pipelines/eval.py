import argparse
import json
import logging
from pathlib import Path
import sys
import typing as tp

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.metrics import calc_metrics

from ml_project.data.make_dataset import prepare_metrics_dict
from ml_project.models.model_fit_predict import predict_model, train_model

PROJECT_PATH = str(Path(__file__).parent.parent)
sys.path.append(PROJECT_PATH)

from ml_project.data import (
    InteractionsTransformer,
    read_data,
    split_data_for_train_test
)
from ml_project.models import evaluate_model
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
    if conf.use_mlflow:
        evaluate_with_mlflow(conf)
    else:
        evaluate(conf)


def evaluate_with_mlflow(conf: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1", type=str)
    version = parser.parse_args().version

    data_url = get_url(
        path=conf.data.output.interactions_path,
        repo=PROJECT_PATH,
        rev=version
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        interactions_df = read_data(
            path=data_url,
            read_params=conf.data.input.interactions.read_params
        )
        
        mlflow.log_param("interactions_version", version)
        mlflow.log_param("interactions_shape", interactions_df.shape)
        
        logger.info(f"Interactions version: {version}")
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

        # mlflow.register_model(model_uri=model_uri, name="RecSysModel")


def evaluate(conf: DictConfig):
    """Evaluate pipeline.

    Args:
        conf (DictConfig): hydra config.
    """
    logger.info("Starting pipeline")

    interactions_df = read_data(
        path=conf.data.output.interactions_path,
        read_params=conf.data.input.interactions.read_params
    )
    logger.info(f"{interactions_df.shape=}")

    transformer = InteractionsTransformer(interactions_column_params=conf.data.input.interactions.column_params)

    interactions_df = transformer.fit_transform(interactions_df)

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

    model, metrics = evaluate_model(
        train_df=train_df,
        test_df=test_df,
        train_params=conf.train_params,
        metric_params=conf.metric_params,
        predict_params=conf.predict_params
    )

    logger.info(f"Metrics is:\n{metrics}")

    with open(conf.data.output.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == "__main__":
    main()
