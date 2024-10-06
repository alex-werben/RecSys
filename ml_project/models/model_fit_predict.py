import pickle
import pandas as pd
import typing as tp
from rectools.models.base import ModelBase
from rectools.models import PureSVDModel
from rectools.metrics import calc_metrics
from rectools.dataset import Dataset
from rectools.columns import Columns
from ml_project.common.predict_params import PredictParams
from ml_project.common.train_params import TrainParams
from ml_project.data.make_dataset import prepare_metrics_dict


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
    dataset: Dataset,
    predict_params: PredictParams
) -> pd.DataFrame:
    """Predict model."""
    interactions_df = dataset.get_raw_interactions()
    users_to_predict = interactions_df[Columns.User].unique()

    recs_df: pd.DataFrame = model.recommend(
        users=users_to_predict,
        dataset=dataset,
        **predict_params
    )

    return recs_df


def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_params: TrainParams,
    metric_params: tp.Dict[str, tp.Dict[str, tp.Any]],
    predict_params: PredictParams
) -> tp.Dict[str, float]:
    """Evaluate model.

    Args:
        train_df (pd.DataFrame): train interactions
        test_df (pd.DataFrame): test interactions
        train_params (TrainParams): train parameters
        metric_params (tp.Dict[str, tp.Dict[str, tp.Any]]): metric parameters
        predict_params (PredictParams): predict parameters

    Returns:
        (tp.Dict[str, float]): dict with metric names and values
    """
    dataset = Dataset.construct(interactions_df=train_df)

    model = train_model(
        dataset=dataset,
        train_params=train_params
    )

    recs_df = model.recommend(
        users=test_df[Columns.User].unique(),
        dataset=dataset,
        **predict_params
    )

    metrics_dict = prepare_metrics_dict(metric_params)

    metrics: tp.Dict[str, float] = calc_metrics(
        metrics=metrics_dict,
        reco=recs_df,
        interactions=test_df,
        prev_interactions=train_df,
    )

    return metrics


def serialize_object(
    obj: tp.Any,
    output: str
) -> None:
    """Save object to output path."""
    with open(output, "wb") as f:
        pickle.dump(obj, f)
