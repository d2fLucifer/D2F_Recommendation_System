import os
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import datetime

dag = DAG(
    dag_id="test_qdrant_connection",
    default_args={
        "owner": "Lucifer ",
        "start_date": airflow.utils.dates.days_ago(1),
    },
    schedule_interval="@daily",
    catchup=False,
)


transform_to_qdrant_job = SparkSubmitOperator(
    task_id="transform_to_qdrant_job",
    conn_id="spark-conn",
    packages="org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.apache.hudi:hudi-spark3.4-bundle_2.12:0.14.0,org.apache.hadoop:hadoop-aws:3.3.2",
    application="jobs/python/transform_to_qdrant.py",
    conf={
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.executor.instances": "1",
        "spark.network.timeout": "800s",
        "spark.executor.heartbeatInterval": "200s"
    },
    dag=dag,
)


transform_to_qdrant_job 
