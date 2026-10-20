FROM python:3.10-slim

# libgomp1 - OpenMP для lightgbm, xgboost, catboost; gcc и g++ - causalml собирается из исходников
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 gcc g++ \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код и данные подключаются томом (-v), в образ не копируются
CMD ["python"]
