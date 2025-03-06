import os
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

from datetime import datetime

default_args = {
    "owner": "Lucifer",
    "start_date": airflow.utils.dates.days_ago(1),
}

dag = DAG(
    dag_id="Inject_data_to_qdrant",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
)

inject_to_qdrant = SparkSubmitOperator(
    task_id="inject_to_qdrant",
    application="/opt/airflow/jobs/python/inject_to_qdrant.py",
    conn_id="spark-conn",
    verbose=True,
    packages="org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,org.apache.hudi:hudi-spark3.4-bundle_2.12:0.14.0,org.apache.hadoop:hadoop-aws:3.3.2",
    conf={
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "spark.executor.instances": "1",
        "spark.network.timeout": "800s",
        "spark.executor.heartbeatInterval": "200s",
    },
    jars="/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar",
    dag=dag,
)

inject_to_qdrant