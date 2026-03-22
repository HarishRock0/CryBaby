FROM apache/airflow:2.10.2-python3.10

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev \
    libsndfile1 \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 -r /requirements.txt --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.10.txt"
