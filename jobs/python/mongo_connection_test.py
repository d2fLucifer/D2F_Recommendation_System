from pyspark.sql import SparkSession
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import VectorParams, OptimizersConfigDiff
from qdrant_client.http.models import Distance
import time
import logging

logger = logging.getLogger(__name__)

def create_spark_session():
    """Initialize Spark session with proper configurations"""
    return SparkSession.builder \
        .appName("MongoToQdrant") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .config("spark.mongodb.input.uri", "mongodb+srv://pnghung2003:pnghung2003@cluster0.xiuaw.mongodb.net/recommendation_system?authSource=admin&ssl=true") \
        .config("spark.mongodb.output.uri", "mongodb+srv://pnghung2003:pnghung2003@cluster0.xiuaw.mongodb.net/recommendation_system?authSource=admin&ssl=true") \
        .config("spark.mongodb.input.sslEnabled", "true") \
        .config("spark.mongodb.output.sslEnabled", "true") \
        .config("spark.mongodb.input.ssl.invalidHostNameAllowed", "true") \
        .config("spark.mongodb.output.ssl.invalidHostNameAllowed", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.auth.login.config=/opt/airflow/jobs/jaas.conf") \
        .config("spark.executor.extraJavaOptions", "-Djava.security.auth.login.config=/opt/airflow/jobs/jaas.conf") \
        .master("local[*]") \
        .getOrCreate()

def setup_qdrant_collection(client, collection_name="product_events"):
    """Initialize Qdrant collection with proper settings"""
    try:
        # Wait for Qdrant to be ready
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            try:
                client.get_collections()
                break
            except Exception as e:
                logger.warning(f"Waiting for Qdrant to be ready (attempt {retry_count + 1}/{max_retries})")
                time.sleep(5)
                retry_count += 1

        logger.info(f"Setting up Qdrant collection: {collection_name}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=0)
        )
        return collection_name
    except Exception as e:
        logger.error(f"Error setting up Qdrant collection: {str(e)}")
        raise 