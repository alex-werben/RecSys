# RecSys
Implementation of some basic recommendation system algorithms.

## Requirements
- Python 3.12.2 or higher

## Installation:
~~~
pip install poetry
poetry install
~~~

## Test:
~~~
poetry pytest tests/
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
