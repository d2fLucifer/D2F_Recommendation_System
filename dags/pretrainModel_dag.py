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
KAFKA_BROKER = "kafka.d2f.io.vn:9092"
TOPIC = "user-behavior-events"
MINIMUM_MESSAGES = 0

# Default arguments for the DAG
default_args = {
    "owner": "Lucifer",
    "start_date": datetime(2025, 3, 9),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def check_kafka_messages():
    """Check if Kafka topic has enough messages, fail if less than minimum"""
    logger.info(f"Starting Kafka message count check for topic: {TOPIC}")
    logger.info(f"Connecting to Kafka broker: {KAFKA_BROKER}")

    try:
        # Initialize Kafka consumer
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BROKER,
            auto_offset_reset='earliest',
            enable_auto_commit=False
        )
        logger.info("Kafka consumer initialized successfully")

        # Get topic partitions
        partitions = consumer.partitions_for_topic(TOPIC)
        if not partitions:
            error_msg = f"Topic '{TOPIC}' does not exist or has no partitions"
            logger.error(error_msg)
            raise ValueError(error_msg)

        total_messages = 0
        
        # Calculate messages for each partition
        for partition in partitions:
            tp = TopicPartition(TOPIC, partition)
            consumer.assign([tp])
            
            consumer.seek_to_beginning(tp)
            earliest_offset = consumer.position(tp)
            
            consumer.seek_to_end(tp)
            latest_offset = consumer.position(tp)
            
            partition_messages = latest_offset - earliest_offset
            total_messages += partition_messages
            
            logger.info(f"Partition {partition}: {partition_messages} messages "
                       f"(offsets {earliest_offset} to {latest_offset})")

        consumer.close()
        logger.info("Kafka consumer closed")
        
        logger.info(f"Total messages found: {total_messages}")
        
        # Check if messages meet minimum requirement
        if total_messages < MINIMUM_MESSAGES:
            error_msg = (f"Task failed: Only {total_messages} messages found, "
                        f"minimum required is {MINIMUM_MESSAGES}")
            logger.error(error_msg)
            raise AirflowException(error_msg)
            
        logger.info("Message count meets minimum requirement")
        return total_messages

    except Exception as e:
        error_msg = f"Error checking Kafka messages: {str(e)}"
        logger.error(error_msg)
        raise AirflowException(error_msg)

# DAG definition
with DAG(
    dag_id="pretrain_model_dag",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    description="DAG to process Kafka stream, inject to Qdrant, and notify AI module",
) as dag:

    # Task 1: Check Kafka message count
    check_kafka_message = PythonOperator(
        task_id="check_kafka_message",
        python_callable=check_kafka_messages,
        trigger_rule="all_success",
    )

    # Task 2: Process Kafka stream
    process_kafka_stream = SparkSubmitOperator(
        task_id="process_kafka_stream",
        conn_id="spark-conn",
        packages=(
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,"
            "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,"
            "org.apache.hudi:hudi-spark3.4-bundle_2.12:0.14.0,"
            "org.apache.hadoop:hadoop-aws:3.3.2"
        ),
        application="jobs/python/process_kafka_stream.py",
        jars="/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar",
        trigger_rule="all_success",
    )

    # Task 3: Inject to Qdrant
    inject_to_qdrant = SparkSubmitOperator(
        task_id="inject_to_qdrant",
        conn_id="spark-conn",
        packages=(
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0,"
            "org.apache.hadoop:hadoop-aws:3.3.2"
        ),
        application="jobs/python/inject_to_qdrant_retrain.py",
        jars="/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar",
        trigger_rule="all_success",
    )

    # Task 4: Notify AI module
    notify_ai_module = SparkSubmitOperator(
        task_id="notify_ai_module",
        conn_id="spark-conn",
        packages="org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0",
        application="jobs/python/notify_ai_module.py",
        trigger_rule="all_success",
    )

    # Set task dependencies
    check_kafka_message >> process_kafka_stream >> inject_to_qdrant >> notify_ai_module