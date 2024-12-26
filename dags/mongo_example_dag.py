from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# Define default_args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'mongo_data_injection_dag',
    default_args=default_args,
    description='DAG to inject data from a CSV file into MongoDB',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    start_task = DummyOperator(task_id='start_task')

    inject_data_to_mongo = SparkSubmitOperator(
        task_id="inject_data_to_mongo",
        conn_id="spark-conn",
        packages="org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.apache.hudi:hudi-spark3.4-bundle_2.12:0.14.0,org.apache.hadoop:hadoop-aws:3.3.2",
        application="jobs/python/load_data_to_mongo.py",  # Update with actual script path
        conf={
            "spark.driver.memory": "4g",
            "spark.executor.memory": "4g",
            "spark.executor.instances": "1",
        },
    )


    end_task = DummyOperator(task_id='end_task')

    start_task >> inject_data_to_mongo >> end_task
