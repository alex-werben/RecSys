import json
import sys
import os
import datetime
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from rectools.columns import Columns
import pandas as pd
import typing as tp

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.data.make_dataset import split_data_for_train_test
from ml_project.data import read_data, process_interactions
from ml_project.models import (
    train_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@hydra.main(version_base="1.1", config_path="../configs", config_name="train_config")
def main(conf: DictConfig = None):
    logger.info("Starting pipeline")

    interactions_df = read_data(conf.data.input.interactions_path)
    logger.info(f"{interactions_df.shape=}")

    interactions_df = process_interactions(interactions_df)

    interactions_df.info()

    train_df, test_df = split_data_for_train_test(interactions_df, conf.splitter_params)

    logger.info(f"{train_df.shape=}")
    logger.info(f"{test_df.shape=}")

    test_df = test_df[test_df[Columns.User].isin(train_df[Columns.User])]

    logger.info(f"{test_df.shape=}")

    metrics = evaluate_model(
        train_df=train_df,
        test_df=test_df,
        train_params=conf.train_params
    )

    logger.info(f"Metrics is:\n{metrics}")

    with open(conf.data.output.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == "__main__":
    main()