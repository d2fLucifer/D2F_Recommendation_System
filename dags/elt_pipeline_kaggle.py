import os
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator


dag = DAG(
    dag_id="ELT_pipeline_kaggle_dataset",
    default_args={
        "owner": "Soumil Shah",
        "start_date": airflow.utils.dates.days_ago(1)
    },
    schedule_interval="@daily"
)



extract_data_job = SparkSubmitOperator(
    task_id="extract_data_job",
    conn_id="spark-conn",
    application="jobs/python/extract_data.py",
    packages="org.apache.hudi:hudi-spark3.4-bundle_2.12:0.14.0,org.apache.hadoop:hadoop-aws:3.3.2",
    conf={
        "spark.driver.memory": "4g",  # Adjust as per your requirement
        "spark.executor.memory": "4g",  # Adjust as per your requirement
        "spark.executor.instances": "1"  # Adjust as per your requirement
    },
    dag=dag
)
load_data_to_minio_job = SparkSubmitOperator(
    task_id="load_data_to_minio_job",
    conn_id="spark-conn",
    application="jobs/python/load_data_to_minio.py",
    dag=dag
)
transform_to_milvus_job = SparkSubmitOperator(
    task_id="transform_to_milvus_job",
    conn_id="spark-conn",
    application="jobs/python/transform_to_milvus.py",
    dag=dag
)

extract_data_job >> load_data_to_minio_job >> transform_to_milvus_job
