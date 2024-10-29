FROM python:3.12.4

RUN python3 -m pip install --upgrade pip setuptools wheel poetry
RUN apt-get update && apt-get install wget -y
RUN poetry config virtualenvs.create false

COPY . ./app
WORKDIR /app

RUN poetry install --no-dev --no-root

ENV MODEL_PATH="models/model.pkl"
ENV BUCKET_NAME="alex-werben-recsys-bucket"

EXPOSE 15000

CMD ["uvicorn", "online_inference.main:app", "--reload", "--host", "0.0.0.0", "--port", "15000"]