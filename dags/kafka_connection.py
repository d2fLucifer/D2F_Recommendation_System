from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
with DAG(
    'spark_kafka_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    # Spark job to process Kafka messages
    spark_task = SparkSubmitOperator(
        task_id='spark_kafka_processing',
        application='/opt/airflow/dags/scripts/spark_kafka_processor.py',
        conn_id='spark_default',
        application_args=['--kafka-bootstrap-servers', 'localhost:9092', '--topic', 'test_topic'],
        executor_cores=2,
        executor_memory='2g',
        driver_memory='1g',
        verbose=True
    )

    spark_task
