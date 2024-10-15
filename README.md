# RecSys
Implementation of some basic recommendation system algorithms.

## Requirements
- Python 3.12.2 or higher

## Installation with Poetry:
~~~
pip install poetry


poetry install
~~~

## Some commands with Poetry
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

## DVC setup:
```
dvc init
# Remote variant
dvc remote add -d <remote_name> s3://<bucket_name>/<dataset_folder>

dvc remote modify <remote_name> access_key_id <your-access-key-id>
dvc remote modify <remote_name> secret_access_key <your-secret-access-key>
dvc remote modify <remote_name> endpointurl https://storage.yandexcloud.net


# Local variant
dvc remote add <local_name> <local_path>

dvc add <file_to_track>
dvc commit
git add .
git commit -m <message>
dvc push -r <local_name>
git push
```

## Test:
~~~
pytest tests/
~~~


## Usage

### Fit model

~~~
python pipelines/trian.py
~~~
After fitting model it'll be saved to /models in .pkl format.

### Predict model

~~~
python pipelines/predict.py
~~~
Process loads model and dataset, after that it makes prediction and saves it.

### Evaluate model

~~~
python pipelines/eval.py
~~~
Process for scoring model, metrics are saved in /models in .json format.

## MLFLow

### Start MLFLow

~~~
mlflow ui
~~~

### Run entry point

~~~
mlflow run . --entry-point <entry_point_name> --env-manager local --experiment-name <exp_name> --run-name <run_name>
~~~
