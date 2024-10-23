from pathlib import Path
import sys
from rectools import Columns
import requests
import logging
import os
from hydra import initialize, compose

sys.path.append(str(Path(__file__).parent.parent.parent))


from ml_project.connections import S3Connector
if not os.path.isdir("logs"):
    os.mkdir("logs")

FORMAT_LOG = "%(asctime)s: %(message)s"
console_out = logging.StreamHandler()

logging.basicConfig(
    handlers=(console_out),
    format=FORMAT_LOG,
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Requests")

response = requests.get(
    "http://0.0.0.0:15000/ready",
)

logger.info(f"Status code: {response.status_code}")
logger.info(f"Message: {response.json()}")

with initialize(config_path="../../configs", version_base=None):
    conf = compose(config_name="train_svd.yaml")
    s3_conn = S3Connector(bucket_name=conf.s3_params.bucket_name)
    recs = s3_conn.get(path=conf.data.output.recommendations_path)

    user_ids = recs[Columns.User].unique()[:3]

    for id in user_ids:
        logger.info(f"User id: {id}")
        response = requests.get(
            url="http://0.0.0.0:15000/predict_for_user",
            params={"id": id}
        )
        logger.info("Response:")
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Message: {response.json()}")
