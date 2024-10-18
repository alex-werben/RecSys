import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from s3 import S3Connector

__all__ = [
    "S3Connector"
]
