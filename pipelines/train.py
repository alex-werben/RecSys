import sys
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from rectools.dataset import Dataset

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.data import read_data, process_interactions
from ml_project.models import (
    train_model,
    serialize_object
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@hydra.main(version_base="1.1", config_path="../configs", config_name="train_config")
def main(conf: DictConfig = None):
    logger.info("Starting pipeline")

    interactions_df = read_data(
        path=conf.data.input.interactions.path,
        read_params=conf.data.input.interactions.read_params
    )
    logger.info(f"{interactions_df.shape=}")

    interactions_column_params = {v: k for k, v in conf.data.input.interactions.column_params.items()}

    interactions_df = process_interactions(
        interactions_df=interactions_df,
        interactions_column_params=interactions_column_params
    )

    logger.info("interactions_df.info():")
    interactions_df.info()

    dataset = Dataset.construct(interactions_df=interactions_df)

    model = train_model(
        dataset=dataset,
        train_params=conf.train_params
    )

    logger.info("Serializing model")
    serialize_object(
        object=model,
        output=conf.data.output.model_path
    )
    
    logger.info("Serializing dataset")
    serialize_object(
        object=dataset,
        output=conf.data.output.dataset_path
    )
    
    logger.info("Pipeline done!")


if __name__ == "__main__":
    main()
