import datetime
import logging
from pathlib import Path
import sys

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from rectools import Columns
from sklearn.preprocessing import LabelEncoder

project_path = str(Path(__file__).parent.parent)
sys.path.append(project_path)

from ml_project.data import read_data, group_interactions, filter_interactions, normalize_weight

load_dotenv()
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@hydra.main(
    version_base="1.1",
    config_path="../configs",
    config_name="train_svd"
)
def preprocess(conf: DictConfig):
    """Predict pipeline.

    Args:
        conf (DictConfig): hydra config.
    """
    logger.info("Starting pipeline")

    interactions_df = read_data(
        path=conf.data.input.interactions.path.initial,
        read_params=conf.data.input.interactions.read_params
    )

    dict_items = conf.data.input.interactions.column_names.items()
    inverse_column_name_mapper = {v: k for k, v in dict_items}
    interactions_df[Columns.Datetime] = datetime.date.today()
    interactions_df = interactions_df.rename(columns=inverse_column_name_mapper)
    le = LabelEncoder()

    interactions_df[Columns.Item] = le.fit_transform(interactions_df[Columns.Item]).astype(int)
    interactions_df[Columns.User] = le.fit_transform(interactions_df[Columns.User]).astype(int)

    interactions_df = group_interactions(interactions_df=interactions_df)

    interactions_df = filter_interactions(interactions_df=interactions_df)

    interactions_df = normalize_weight(interactions_df=interactions_df)

    interactions_df.to_csv(conf.data.input.interactions.path.processed, index=False)


if __name__ == "__main__":
    preprocess()
