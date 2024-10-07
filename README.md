# RecSys
Implementation of some basic recommendation system algorithms.

## Requirements
- Python 3.12.2 or higher

## Installation:
~~~
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
~~~

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