from airflow import DAG
from airflow.operators.python import PythonOperator
from qdrant_client import QdrantClient
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qdrant_connection():
    """Main function to test Qdrant connection."""
    try:
        client = QdrantClient(url="http://qdrant:6333")
        logger.info("Connected to Qdrant")

        # List all collections
        collections_response = client.get_collections()
        collections = collections_response.collections if hasattr(collections_response, 'collections') else []
        collection_names = [collection.name for collection in collections]
        logger.info(f"Existing collections: {collection_names}")

        # Check if 'my_collection' exists
        collection_name = "my_collection"
        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' does not exist. Creating it now...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": 128, "distance": "Cosine"}  # Adjust as needed
            )
            logger.info(f"Collection '{collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

        # Fetch collection details
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection info: {collection_info}")
    except Exception as e:
        logger.error(f"Error while testing Qdrant connection: {e}")
        raise

# Define the Airflow DAG
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="test_qdrant_connection_dag",
    default_args=default_args,
    start_date=datetime(2024, 12, 24),
    schedule_interval=None,  # Run manually or as needed
    catchup=False,
) as dag:

    test_qdrant_task = PythonOperator(
        task_id="test_qdrant_connection",
        python_callable=test_qdrant_connection,
    )

    test_qdrant_task
