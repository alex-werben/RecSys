import pickle
import pandas as pd
import numpy as np
import typing as tp
from rectools.models.base import ModelBase
from rectools.models import PureSVDModel
from rectools.metrics import calc_metrics, Recall, Precision
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

def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_params: TrainParams
):
    dataset = Dataset.construct(interactions_df=train_df)

    model = train_model(
        dataset=dataset,
        train_params=train_params
    )

    recs_df = model.recommend(
        users=test_df[Columns.User].unique(),
        dataset=dataset,
        k=10,       # TODO: add eval params
        filter_viewed=True
    )

    metrics_dict = {
        "Recall_10": Recall(k=10),  # TODO: configure
        "Precision_10": Precision(k=10)
    }

    metrics: tp.Dict[str, float] = calc_metrics(
        metrics=metrics_dict,
        reco=recs_df,
        interactions=test_df,
        prev_interactions=train_df,
    )

    return metrics

def serialize_model(
    model: ModelBase,
    output: str
) -> None:
    """Saves model to output path."""
    with open(output, "wb") as f:
        pickle.dump(model, f)
