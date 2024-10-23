import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rectools import Columns

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI
from ml_project.connections import S3Connector
from hydra import initialize, compose
from pipelines import predict

load_dotenv()
app = FastAPI()


@app.get('/')
def home():
    return "Hello, world!"


@app.get("/ready")
def check_ready():
    s3_conn = S3Connector(bucket_name=os.getenv("BUCKET_NAME"))
    model_exists = s3_conn.check_file_exists(path=os.getenv("MODEL_PATH"))

    if model_exists:
        return "Model is ready."
    else:
        return "Model is not ready."


@app.get("/predict")
def predict_route():
    with initialize(config_path="../configs", version_base=None):
        conf = compose(config_name="train_svd.yaml")

        predict(conf)


@app.get("/predict_for_user")
async def predict_for_user_route(id: int):
    with initialize(config_path="../configs", version_base=None):
        conf = compose(config_name="train_svd.yaml")

        s3_conn = S3Connector(bucket_name=conf.s3_params.bucket_name)

        recs_df = s3_conn.get(path=conf.data.output.recommendations_path)

        user_recs = recs_df[recs_df[Columns.User] == id].to_html()

        return user_recs
