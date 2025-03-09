# Path: dags/pretrain_model_dag.py
from airflow import DAG
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
    'pretrainModel_dag',
    default_args=default_args,
    description='DAG to process Kafka data and pretrain model',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    start_task = DummyOperator(task_id='start_task')

    # Process Kafka stream and save to temporary storage
    process_kafka_stream = SparkSubmitOperator(
        task_id="process_kafka_stream",
        conn_id="spark-conn",
        packages="org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,"
                "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,"
                "org.apache.hudi:hudi-spark3.4-bundle_2.12:0.14.0,"
                "org.apache.hadoop:hadoop-aws:3.3.2",
        application="jobs/python/pretrainModel_extract.py",
        jars ="/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar",
    )

 

    end_task = DummyOperator(task_id='end_task')

    start_task >> process_kafka_stream  >> end_task