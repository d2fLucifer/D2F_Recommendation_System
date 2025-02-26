from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

# Define default_args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    'test_package_dag',
    default_args=default_args,
    description='DAG to inject data from a CSV file into MongoDB and Qdrant',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    start_task = DummyOperator(task_id='start_task')

    test_package_job = SparkSubmitOperator(
        task_id="test_package_job",
        conn_id="spark-conn",
        application="jobs/python/processing_data.py",
        name="test_package_job",
        packages="org.mongodb.spark:mongo-spark-connector_2.12:3.0.1",
        jars ="/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar",
        conf={
            "spark.driver.memory": "4g",
            "spark.executor.memory": "4g",
            "spark.executor.instances": "1",
            # Additional Spark configurations if needed
        },
      
    )



    end_task = DummyOperator(task_id='end_task')

    start_task >> test_package_job >> end_task
