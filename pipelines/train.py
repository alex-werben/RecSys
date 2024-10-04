import sys
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from rectools.columns import Columns
from rectools.dataset import Dataset
import pandas as pd
import datetime

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.data import read_data, process_interactions
from ml_project.models import (
    train_model,
    evaluate_model,
    serialize_model
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

    dataset = Dataset.construct(interactions_df=interactions_df)

    model = train_model(
        dataset=dataset,
        train_params=conf.train_params
    )

    serialize_model(
        model=model,
        output=conf.data.output.model_path
    )


if __name__ == "__main__":
    main()