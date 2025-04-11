# spark_submit.py

# Core Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws
from pyspark.sql.types import StringType, ArrayType, FloatType
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.linalg import Vectors

# Standard Python imports
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# External library imports
from qdrant_client import QdrantClient

# Custom imports
from spark_session import create_spark_session

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Main Execution Logic
if __name__ == "__main__":
    # Create Spark session with optimized configurations
    spark = create_spark_session()
    logger.info("Spark session created successfully")

    # Read data from S3 without immediate caching
    s3_dataset_path = os.getenv("S3_DATASET_PATH")
    df = spark.read \
        .option("header", "true") \
        .option("mode", "PERMISSIVE") \
        .option("columnNameOfCorruptRecord", "_corrupt_record") \
        .csv(s3_dataset_path)

    logger.info(f"Successfully read CSV file from {s3_dataset_path}")
    df.show(5)  # Limit rows for quick inspection
    logger.info(f"Total records: {df.count()}")

    # Combine attributes efficiently
    df_combined = df.withColumn(
        "combined_attributes",
        concat_ws(" ",
            col("user_id"),
            col("product_id"),
            col("name"),
            col("brand"),
            col("category_code"),
            col("price"),
            col("event_type"),
            col("event_time"),
            col("user_session")
        )
    ).select("product_id", "combined_attributes")
    df.unpersist()  # Free memory if cached implicitly

    # NLP Feature Engineering
    tokenizer = Tokenizer(inputCol="combined_attributes", outputCol="words")
    df_tokenized = tokenizer.transform(df_combined)

    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_filtered = stopwords_remover.transform(df_tokenized).cache()  # Cache here for Word2Vec
    df_tokenized.unpersist()  # Free memory

    word2Vec = Word2Vec(
        vectorSize=128,  # Reduced from 128
        minCount=10,    # Increased from 5
        inputCol="filtered_words",
        outputCol="word2vec_features",
        numPartitions=200
    )
    model = word2Vec.fit(df_filtered)
    df_vectorized = model.transform(df_filtered)

    # Convert VectorUDT to ArrayType for Qdrant compatibility
    def vector_to_array(vector):
        return vector.toArray().tolist()

    vector_to_array_udf = udf(vector_to_array, ArrayType(FloatType()))

    # Prepare final DataFrame
    df_final = df_vectorized.select(
        "product_id",
        vector_to_array_udf(col("word2vec_features")).alias("vector")
    ).repartition(200, "product_id")  # Repartition for write efficiency

    # Verify schema (optional, for debugging)
    df_final.printSchema()

    # Define COLLECTION_NAME from .env
    COLLECTION_NAME = os.getenv("QDRANT_COMBINED_VECTOR_COLLECTION")

    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL")
    client = QdrantClient(url=qdrant_url)

    # Create Qdrant collection (run only once, ideally outside the script)
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": 128, "distance": "Cosine"},  # Match vectorSize
        )
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' created or recreated.")
    except Exception as e:
        logger.error(f"Failed to recreate Qdrant collection: {e}")

    # Configure connection options for Qdrant
    qdrant_grpc_url = os.getenv("QDRANT_GRPC_URL")
    options = {
        "qdrant_url": qdrant_grpc_url,  # Use GRPC URL for writing
        "collection_name": COLLECTION_NAME,
        "embedding_field": "vector",
        "schema": df_final.schema.json(),
    }

    # Write DataFrame to Qdrant with batching
    df_final.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .save()
    logger.info("✅ Data successfully inserted into Qdrant.")

    # Clean up memory
    df_filtered.unpersist()
    spark.stop()
    logger.info("Spark session stopped.")