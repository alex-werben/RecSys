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
