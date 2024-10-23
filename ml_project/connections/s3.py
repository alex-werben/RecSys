import boto3
import typing as tp
import os
from io import BytesIO
from botocore.exceptions import ClientError
import pickle


class S3Connector:
    """Wrapper for S3 connection."""

    ENDPOINT: str = "https://storage.yandexcloud.net"
    REGION: str = "ru-central1"
    SUCCESS: int = 200

    def __init__(
        self,
        bucket_name: str,
    ) -> None:
        """Initialize S3 connector.

        Args:
            bucket_name (str): bucket name.
        """
        self.bucket_name = bucket_name

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
        """Put object to S3.

        Args:
            obj (tp.Any): object to save
            path (str): path where it'll be located

        Returns:
            int: status code
        """
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
        """Get object from S3.

        Args:
            path (str): path where object is located in S3

        Returns:
            tp.Any: object from S3 or status code if error occurs
        """
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

    def check_file_exists(self, path: str) -> bool:
        """Check existence of file in S3.

        Args:
            path (str): path to file

        Returns:
            bool: whether file exists or not
        """
        try:
            self.s3_conn.head_object(
                Bucket=self.bucket_name,
                Key=path
            )
            return True
        except ClientError as e:
            # Check if the exception is because the file does not exist
            if e.response['Error']['Code'] == '404':
                return False  # File not found
            else:
                # Something else went wrong
                raise e
