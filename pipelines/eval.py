import json
import logging
from pathlib import Path
import sys

import hydra
import mlflow
from omegaconf import DictConfig
from rectools.columns import Columns

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.data import (
    InteractionsTransformer,
    read_data,
    split_data_for_train_test
)
from ml_project.models import evaluate_model

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
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        interactions_df = read_data(
            path=conf.data.input.interactions.path,
            read_params=conf.data.input.interactions.read_params
        )
        logger.info(f"{interactions_df.shape=}")

        transformer = InteractionsTransformer(interactions_column_params=conf.data.input.interactions.column_params)

        interactions_df = transformer.fit_transform(interactions_df)
        
        mlflow.log_artifact(
            local_path=conf.data.input.interactions.path,
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

        model, metrics = evaluate_model(
            train_df=train_df,
            test_df=test_df,
            train_params=conf.train_params,
            metric_params=conf.metric_params,
            predict_params=conf.predict_params
        )

        model_uri = f"runs:/{run_id}/recsys"
        mlflow.log_metrics(metrics=metrics)

        mlflow.register_model(model_uri=model_uri, name="RecSysModel")
        
    

def evaluate(conf: DictConfig):
    """Evaluate pipeline.

    Args:
        conf (DictConfig): hydra config.
    """
    logger.info("Starting pipeline")

    interactions_df = read_data(
        path=conf.data.input.interactions.path,
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
