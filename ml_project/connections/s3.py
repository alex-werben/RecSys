import boto3
import typing as tp
import os
from io import BytesIO
import pickle

class S3Connector:
    ENDPOINT: str = "https://storage.yandexcloud.net"
    REGION: str = "ru-central1"
    SUCCESS: int = 200
    
    def __init__(
        self,
        bucket_name: str,
    ) -> None:
        self.bucket_name = bucket_name

        print(os.getenv("S3_ACCESS_KEY"))
        print(os.getenv("S3_SECRET_KEY"))

        session = boto3.Session(
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            region_name=self.REGION,
        )
        
        self.s3_conn = session.client(
            "s3", endpoint_url=self.ENDPOINT
        )
    
    def put(
        self,
        obj: tp.Any,
        path: str
    ) -> int:
        buffer = BytesIO()
        pickle.dump(obj, buffer)

        response = self.s3_conn.put_object(
            Body=buffer.getvalue(),
            Bucket=self.bucket_name,
            Key=path
        )

        status_code = response["ResponseMetadata"]["HTTPStatusCode"]

        return status_code

    def get(
        self,
        path: str
    ) -> tp.Any:
        response = self.s3_conn.get_object(
            Bucket=self.bucket_name,
            Key=path
        )
        
        status_code = response["ResponseMetadata"]["HTTPStatusCode"]
        
        if status_code == self.SUCCESS:
            obj = pickle.load(response["Body"])
            return obj
        else:
            return status_code
