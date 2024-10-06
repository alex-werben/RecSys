import logging
from pathlib import Path
import pickle
import sys

import hydra
from omegaconf import DictConfig

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.models import predict_model

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
    """Predict pipeline.

    Args:
        conf (DictConfig): hydra config.
    """
    logger.info("Starting pipeline")

    logger.info(f"Loading model from {conf.data.output.model_path}")
    with open(conf.data.output.model_path, "rb") as model_file:
        model = pickle.load(model_file)
    logger.info("Model loaded successfully")

    logger.info(f"Loading dataset from {conf.data.output.dataset_path}")
    with open(conf.data.output.dataset_path, "rb") as dataset_file:
        dataset = pickle.load(dataset_file)
    logger.info("Dataset loaded successfully")

    logger.info("Predicting recommendations")
    recs_df = predict_model(
        model=model,
        dataset=dataset,
        predict_params=conf.predict_params
    )

    recs_df.info()

    recs_df.to_csv(conf.data.output.recommendations_path, index=False)


if __name__ == "__main__":
    main()