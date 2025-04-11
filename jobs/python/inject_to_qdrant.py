# process_to_qdrant.py

# Core Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, monotonically_increasing_id
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, LongType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Standard Python imports
import logging
from datetime import datetime
import numpy as np
import os
from dotenv import load_dotenv

# External library imports
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Custom imports
from spark_session import create_spark_session

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 500000  
COLLECTION_NAME = os.getenv("QDRANT_TEST_COLLECTION")

def process_chunk(spark, df_chunk, chunk_id, embedding_model, client, options):
    """Process a single chunk of data and upsert to Qdrant."""
    logger.info(f"Processing chunk {chunk_id} with {df_chunk.count()} records")

    # NLP Feature Engineering
    tokenizer = Tokenizer(inputCol="name", outputCol="words")
    df_tokenized = tokenizer.transform(df_chunk)

    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_filtered = stopwords_remover.transform(df_tokenized)

    # Collect filtered words for embedding
    filtered_words = df_filtered.select("filtered_words").collect()
    documents = [" ".join(row["filtered_words"]) for row in filtered_words]

    # Generate embeddings
    embeddings = list(embedding_model.embed(documents))
    embeddings = [emb.astype(np.float32).tolist()[:128] for emb in embeddings]  # Truncate to 128 dimensions

    # Define schema for embeddings
    schema = StructType([
        StructField("id", LongType(), False),
        StructField("vector", ArrayType(FloatType()), False)
    ])

    # Add embeddings back to DataFrame
    df_filtered = df_filtered.withColumn("id", monotonically_increasing_id())
    embeddings_df = spark.createDataFrame([(i, emb) for i, emb in enumerate(embeddings)], schema=schema)
    df_final = df_filtered.join(embeddings_df, "id").drop("id")

    # Select columns matching the Qdrant schema, including vector
    df_final = df_final.select(
        "event_time", "event_type", "product_id", "category_code",
        "brand", "price", "user_id", "user_session", "name", "vector"
    )

    # Drop duplicates within chunk
    df_final = df_final.dropDuplicates()

    # Write to Qdrant
    df_final.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .save()

    logger.info(f"Chunk {chunk_id} successfully inserted into Qdrant.")
    # Explicitly unpersist to free memory
    df_final.unpersist()
    df_filtered.unpersist()
    df_tokenized.unpersist()

def main():
    # Create Spark session
    spark = create_spark_session()

    # Read CSV file
    s3_dataset_path = os.getenv("S3_DATASET_PATH")
    df = spark.read \
        .option("header", "true") \
        .option("mode", "PERMISSIVE") \
        .option("columnNameOfCorruptRecord", "_corrupt_record") \
        .csv(s3_dataset_path)
    
    # df = df.limit(10000)  # Commented out as in original

    df.printSchema()

    logger.info(f"Successfully read CSV file from {s3_dataset_path}")
    total_count = df.count()
    logger.info(f"Total records: {total_count}")

    # Initialize fastembed TextEmbedding
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_ALTERNATE_URL")
    client = QdrantClient(url=qdrant_url)

    # Create Qdrant collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 128, "distance": "Cosine"},
    )

    # Define the schema for Qdrant, including vector as an array of floats
    qdrant_schema = StructType([
        StructField("event_time", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("category_code", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("price", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("user_session", StringType(), True),
        StructField("name", StringType(), True),
        StructField("vector", ArrayType(FloatType()), True)  # Vector as array of floats
    ])

    # Qdrant connection options
    qdrant_grpc_url = os.getenv("QDRANT_ALTERNATE_GRPC_URL")
    options = {
        "qdrant_url": qdrant_grpc_url,  # Use GRPC port
        "collection_name": COLLECTION_NAME,
        "embedding_field": "vector",  # Still specify vector as embedding field
        "schema": qdrant_schema.json(),
    }

    # Calculate number of chunks
    num_chunks = (total_count + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(f"Total chunks to process: {num_chunks}")

    # Process in chunks
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * CHUNK_SIZE
        end_idx = min((chunk_id + 1) * CHUNK_SIZE, total_count)
        
        # Get chunk of data
        df_chunk = df.limit(end_idx).subtract(df.limit(start_idx))
        df_chunk.cache()  # Cache chunk to optimize processing
        
        process_chunk(spark, df_chunk, chunk_id, embedding_model, client, options)
        
        # Clear memory
        df_chunk.unpersist()
        spark.catalog.clearCache()  # Clear Spark cache

    # Stop Spark
    spark.stop()
    logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()