from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
from kafka import KafkaConsumer, TopicPartition
from airflow.exceptions import AirflowException
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
KAFKA_BROKER = "kafka:29092"
TOPIC = "user-behavior-events"
MINIMUM_MESSAGES = 0

# Default arguments for the DAG
default_args = {
    "owner": "Lucifer",
    "start_date": datetime(2025, 3, 9),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
with DAG(
    dag_id="pretrain_model_dag",
    default_args=default_args,
    schedule_interval="0 0 27 * *",  # Run at 00:00 on the 27th of every month
    catchup=False,
    description="DAG to process Kafka stream, inject to Qdrant, and notify AI module",
) as dag:

    # Task: Notify AI module
    notify_ai_module = SparkSubmitOperator(
        task_id="notify_ai_module",
        conn_id="spark-conn",
        packages="org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0",
        application="jobs/python/notify_ai_module.py",
        trigger_rule="all_success",
    )

    # Set task dependencies (if there are others, add here)
    notify_ai_module
