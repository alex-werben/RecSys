import pickle
import pandas as pd
import numpy as np
from rectools.models.base import ModelBase
from rectools.models import PureSVDModel
from rectools.dataset import Dataset
from rectools.columns import Columns
from ml_project.common.train_params import TrainParams

def train_model(
    dataset: Dataset,
    train_params: TrainParams,
) -> ModelBase:
    """Train model."""
    if train_params.model_type == "SVD":
        model = PureSVDModel()

    model.fit(dataset)

    return model

def predict_model(
    model: ModelBase,
    dataset: Dataset
) -> pd.DataFrame:
    """Predict model."""
    interactions_df = dataset.get_raw_interactions()
    users_to_predict = interactions_df[Columns.User].unique()

    recs_df: pd.DataFrame = model.recommend(
        users=users_to_predict,
        dataset=dataset,
        k=10,
        filter_viewed=True
    )

    return recs_df

def evaluate_model():
    pass

def serialize_model(
    model: ModelBase,
    output: str
) -> None:
    """Saves model to output path."""
    with open(output, "wb") as f:
        pickle.dump(model, f)
