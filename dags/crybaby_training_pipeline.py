"""
DAG for training and logging the CryBaby model using MLflow.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

def train_model():
    os.system('jupyter nbconvert --to notebook --execute model.ipynb --output model_output.ipynb')

def log_model():
    os.system('python -c "import mlflow; mlflow.set_experiment(\'CryBaby-Model\')"')

def preprocess_data():
    os.system('python preprocessing.py')

def tune_hyperparameters():
    os.system('python hyperparameter_tuning.py')

def main():
    preprocess_data()
    tune_hyperparameters()
    train_model()
    log_model()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 3, 21),
    'retries': 1
}

dag = DAG(
    dag_id='crybaby_training_pipeline',
    default_args=default_args,
    description='CryBaby ML pipeline with preprocessing, tuning, training, and logging',
    schedule=None,
    catchup=False
)

preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

tune = PythonOperator(
    task_id='tune_hyperparameters',
    python_callable=tune_hyperparameters,
    dag=dag
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

log = PythonOperator(
    task_id='log_model',
    python_callable=log_model,
    dag=dag
)

preprocess >> tune >> train >> log
