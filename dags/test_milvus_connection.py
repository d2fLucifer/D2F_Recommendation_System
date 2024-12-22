from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your_email@example.com'],  # Update with your email
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'milvus_connection_test',
    default_args=default_args,
    description='A DAG to test connection to Milvus using Airflow',
    schedule_interval=None,  # Set to None for manual trigger
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['milvus', 'connection', 'test'],
) as dag:

    def test_milvus_connection():
        """
        Function to test connection to Milvus.
        """
        try:
            # Retrieve connection details from Airflow Connections
            # Replace 'milvus_default' with your actual connection ID
            conn = BaseHook.get_connection("milvus_default")
            
            host = conn.host
            port = conn.port

            # Log connection details (avoid logging sensitive info)
            logging.info(f"Attempting to connect to Milvus at {host}:{port}")

            # Establish connection to Milvus
            connections.connect(
                alias="default",
                host=host,
                port=port,
                timeout=10  # seconds
            )
            logging.info(f"Successfully connected to Milvus at {host}:{port}")

            # Perform a simple operation: list existing collections
            collections = utility.list_collections()
            logging.info(f"Existing collections in Milvus: {collections}")

            # Optionally, create and drop a test collection to further verify
            test_collection_name = "test_airflow_connection"

            if test_collection_name not in collections:
                # Define the schema for the test collection
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
                ]
                schema = CollectionSchema(fields=fields, description="Test collection for Airflow connection")

                # Create the collection using the Collection class
                collection = Collection(name=test_collection_name, schema=schema)
                logging.info(f"Created test collection '{test_collection_name}'")

            # Clean up: Drop the test collection if it was created
            if test_collection_name in utility.list_collections():
                collection = Collection(name=test_collection_name)
                collection.drop()
                logging.info(f"Dropped test collection '{test_collection_name}'")

            # Disconnect after operations
            connections.disconnect(alias="default")
            logging.info("Disconnected from Milvus")

        except Exception as e:
            logging.error(f"Failed to connect to Milvus: {e}")
            raise

    # Define the task
    milvus_connection_test = PythonOperator(
        task_id='test_milvus_connection',
        python_callable=test_milvus_connection,
    )

    milvus_connection_test
