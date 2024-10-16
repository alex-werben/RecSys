import sys
import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
from rectools.dataset import Dataset
from dotenv import load_dotenv
import mlflow

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.connections import S3Connector
from ml_project.data import read_data, InteractionsTransformer
from ml_project.models import train_model

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
    """Train pipeline.

    Args:
        conf (DictConfig): hydra config.
    """
    load_dotenv()

    logger.info("Starting pipeline")

    s3_conn = S3Connector(
        bucket_name=conf.s3_params.bucket_name
    )

    interactions_df = read_data(
        path=conf.data.input.interactions.path,
        read_params=conf.data.input.interactions.read_params
    )
    logger.info(f"{interactions_df.shape=}")

    transformer = InteractionsTransformer(interactions_column_params=conf.data.input.interactions.column_params)

    interactions_df = transformer.fit_transform(interactions_df)

    logger.info("interactions_df.info():")
    interactions_df.info()

    dataset = Dataset.construct(interactions_df=interactions_df)

    model = train_model(
        dataset=dataset,
        train_params=conf.train_params
    )

    logger.info("Saving model")
    s3_conn.put(
        obj=model,
        path=conf.data.output.model_path
    )

    logger.info("Saving dataset")
    s3_conn.put(
        obj=dataset,
        path=conf.data.output.dataset_path
    )

    logger.info("Pipeline done!")


if __name__ == "__main__":
    main()
