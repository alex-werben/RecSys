import logging
from pathlib import Path
import sys

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.models import predict_model
from ml_project.connections import S3Connector

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
load_dotenv()


@hydra.main(
    version_base="1.1",
    config_path="../configs",
    config_name="train_svd"
)
def predict(conf: DictConfig):
    """Predict pipeline.

    Args:
        conf (DictConfig): hydra config.
    """
    logger.info("Starting pipeline")
    s3_conn = S3Connector(
        bucket_name=conf.s3_params.bucket_name
    )

    logger.info(f"Loading model from {conf.data.output.model_path}")
    model = s3_conn.get(path=conf.data.output.model_path)
    logger.info("Model loaded successfully")

    logger.info(f"Loading dataset from {conf.data.output.dataset_path}")
    dataset = s3_conn.get(path=conf.data.output.dataset_path)
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
    predict()
