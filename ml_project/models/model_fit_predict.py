import multiprocessing
import pickle
import pandas as pd
import typing as tp
from rectools.models.base import ModelBase
from rectools.models import (
    PureSVDModel,
    ImplicitALSWrapperModel
)
from implicit.als import AlternatingLeastSquares
from rectools.metrics import calc_metrics
from rectools.dataset import Dataset
from rectools.columns import Columns
from ml_project.common import (
    MetricParams,
    PredictParams,
    TrainParams
)
from ml_project.data.make_dataset import prepare_metrics_dict

MAX_CPU = multiprocessing.cpu_count()


def train_model(
    dataset: Dataset,
    train_params: TrainParams,
) -> ModelBase:
    """Train model."""

    if train_params.model_type == "PureSVDModel":
        model = PureSVDModel(**train_params.model_params)
    elif train_params.model_type == "ImplicitALSWrapperModel":
        model = ImplicitALSWrapperModel(
            model=AlternatingLeastSquares(
                **train_params.model_params,
                num_threads=MAX_CPU
            ),
            verbose=1
        )

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
    metric_params: MetricParams,
    predict_params: PredictParams
) -> tp.Tuple[ModelBase, tp.Dict[str, float]]:
    """Evaluate model.

    Args:
        train_df (pd.DataFrame): train interactions
        test_df (pd.DataFrame): test interactions
        train_params (TrainParams): train parameters
        metric_params (MetricParams): metric parameters
        predict_params (PredictParams): predict parameters

    Returns:
        (Tuple[ModelBase, tp.Dict[str, float]]): model and dict with metric names and values
    """
    dataset = Dataset.construct(interactions_df=train_df)

    if train_params.model_type == "SVD":
        model = PureSVDModel()

    model.fit(dataset)

    users_to_predict = test_df[Columns.User].unique()

    recs_df = model.recommend(
        users=users_to_predict,
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

    return model, metrics


def serialize_object(
    obj: tp.Any,
    path: str
) -> None:
    """Save object to output path."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
