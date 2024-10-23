import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI

from ml_project.connections import S3Connector

load_dotenv()
app = FastAPI()

@app.get('/')
def home():
    return {"key": "Hello, world!"}

async def success_handler(**kwargs):
    return "Model is ready"


async def failure_handler(**kwargs):
    return "Model is not ready"

@app.get("/ready")
def check_ready():
    s3_conn = S3Connector(bucket_name=os.getenv("BUCKET_NAME"))
    model_exists = s3_conn.check_file_exists(path=os.getenv("MODEL_PATH"))
    
    if model_exists:
        return "Model is ready."
    else:
        return "Model is not ready."
    
