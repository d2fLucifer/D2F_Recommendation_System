import os
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import datetime

dag = DAG(
    dag_id="ELT_pipeline_kaggle_dataset",
    default_args={
        "owner": "Lucifer ",
        "start_date": airflow.utils.dates.days_ago(1),
    },
    schedule_interval="@daily",
    catchup=False,
)

extract_data_job = SparkSubmitOperator(
    task_id="extract_data_job",
    conn_id="spark-conn",
    application="jobs/python/extract_data.py",
    packages="org.apache.hudi:hudi-spark3.4-bundle_2.12:0.14.0,org.apache.hadoop:hadoop-aws:3.3.2",
    conf={
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.executor.instances": "1",
    },
    dag=dag,
)

load_data_to_minio_job = SparkSubmitOperator(
    task_id="load_data_to_minio_job",
    conn_id="spark-conn",
    application="jobs/python/load_data_to_minio.py",
    packages="org.apache.hadoop:hadoop-aws:3.3.2",  # Ensure necessary packages
    conf={
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.executor.instances": "1",
    },
    dag=dag,
)

transform_to_milvus_job = SparkSubmitOperator(
    task_id="transform_to_milvus_job",
    conn_id="spark-conn",
    application="jobs/python/transform_to_milvus.py",
    packages="org.apache.hadoop:hadoop-aws:3.3.2",  # Corrected
    conf={
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.executor.instances": "1",
    },
    dag=dag,
)


# Define task dependencies
extract_data_job >> load_data_to_minio_job >> transform_to_milvus_job
