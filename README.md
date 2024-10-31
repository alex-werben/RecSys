# Recommender System
MLOps project, production-ready recommender system.

## Preparation and installation
### Requirements:

- Python 3.12.2 or higher
- Docker desktop

### Installation:
```
pip install poetry

poetry install
```

## Usage
There're 4 main stages: 
1. **preprocess** - loads raw interactions, processes them and saves;
2. **fit** - loads processed interactions, prepares dataset, fits model; saves model and dataset to S3 storage;
3. **predict** - loads model and dataset from S3, makes predictions and saves them;
4. **evaluate** - loads processed interactions, splits them into train and test data, fits model, predicts recommendations, calculates metrics.


### With python and dvc
Each stage can be called separately:
```
python pipelines/preprocess.py
python pipelines/train.py
python pipelines/predict.py
python pipelines/eval.py
```

There's a DVC pipeline that allows to run all stages in right order. It's defined in `/dvc.yaml`. To start pipeline run the following command:
```
dvc repro
```

There are a few unit tests to check some modules and one integration test to check stage of pipeline. To run all tests use the following command:
```
pytest tests/
```

### With docker image and FastAPI

#### Docker setup
##### Create your own image
```
# Build image
docker build -t <username>/<image_name>:<version>
```

##### Pull prepared image from Docker Hub
```
docker pull alexwerben/recsys:v3
```

##### Run container

```
# Run container
docker run --name <container_name> --env-file .env -p 15000:15000 -d <username>/<image_name>:<version>

# Get inside running container shell
docker exec -it <container_name> bash
```
- *There must be `.env` file in project dir with access and secret keys to connect to S3.*

#### FastAPI

##### End points
These frameworks are used to setup backend server with some end points to access service. End points can be accessed after running container, they are:

- 0.0.0.0:15000/ - start page, should output "Hello, world!";
- 0.0.0.0:15000/ready - check if model exists in S3 storage and is ready for usage;
- 0.0.0.0:15000/predict - downloads model from S3, makes prediction, saves recommendations;
- 0.0.0.0:15000/predict_for_user?id={id} - gets recommendations for user with `user_id=={id}` and returns them in html format.

##### Request script

There's also a FastAPI script that makes multiple requests to get recommendations for different users, and checks if model is ready. It can be done by running the following command:
```
python online_inference/requests/make_request.py
```

## Project structure
```
.
├── .dvc                               <- DVC configuration
│   └── config
├── .dvcignore                         <- DVC ignore files
├── .github
│   └── workflows                      <- CI github files
├── .gitignore
├── MLproject                          <- Entry points for MLFlow
├── README.md
├── configs                            <- configuration files for pipelines
├── data
│   ├── processed                      <- processed data
│   └── raw                            <- raw data
├── dockerfile                         <- docker config file
├── dvc.lock
├── dvc.yaml                           <- DVC pipeline file with stages
├── LICENSE                            <- Apache license file
├── ml_project
│   ├── common                         <- dataclasses for params from config
│   ├── connections                    <- connectors to storages
│   ├── data                           <- data processing module
│   ├── models                         <- choose model/fit/predict/eval
├── models                             <- saves model here
├── notebooks                           
├── outputs                            <- hydra outputs
├── pipelines                          <- all pipelines are defines here
├── poetry.lock
├── pyproject.toml                     <- project file with all dependencies
└── tests
    ├── conftest.py                    <- mocking vars and methods for test
    ├── data                           <- unit tests for data module
    │   └── test_make_dataset.py
    ├── dataset_example.csv            <- example data for tests
    └── test_train.py                  <- integration test
```

## Additional instructions

### Poetry usage
```
# If you want to create .venv in project dir, run command below, else skip it
poetry config virtualenvs.in-project true

# If you want to run scripts by "python <script_name.py>", run command below,
# or you'll need to run it by "poetry run python <script_name.py>"
poetry shell

# You can check which python interpreter is used by running:
which python

# To deactivate the virtual environment and exit new shell:
exit

# To deactivate the virtual environment without leaving new shell:
deactivate
```

### DVC usage

#### Setup
```
dvc init
# Remote variant
dvc remote add -d <remote_name> s3://<bucket_name>/<dataset_folder>

dvc remote modify <remote_name> access_key_id <your-access-key-id>
dvc remote modify <remote_name> secret_access_key <your-secret-access-key>
dvc remote modify <remote_name> endpointurl https://storage.yandexcloud.net

# Local variant
dvc remote add <local_name> <local_path>
```

#### Versioning data
```
dvc add <file_to_track>
dvc push -r <local_name>
git add .
git commit -m <message>
git tag -a <tag-name> -m <tag-message>
git push
```

#### Run DVC pipeline
```
dvc repro
```

### MLFLow usage

#### Start MLFLow UI

```
mlflow ui
```

#### Run entry point

```
mlflow run . -e <entry_point_name> --env-manager local --experiment-name <exp_name> --run-name <run_name> -P <param-name>=<param-value>
```
