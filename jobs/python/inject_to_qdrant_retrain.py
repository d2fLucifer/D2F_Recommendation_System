# Standard Python imports
import os
import logging
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, concat_ws, udf, date_format, lit
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, TimestampType
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from spark_session import create_spark_session
from dotenv import load_dotenv

# External library imports
from fastembed import TextEmbedding

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Spark session
logger.info("Creating Spark session...")
spark = create_spark_session(app_name="QdrantInsert")

try:
    # Read data from S3
    s3_pretrain_data_path = os.getenv("S3_PRETRAIN_DATA_PATH")
    logger.info(f"Reading data from S3: {s3_pretrain_data_path}...")
    df = spark.read.csv(s3_pretrain_data_path, header=True)
    df.show(5)

    # NLP Feature Engineering
    logger.info("Starting NLP feature engineering...")
    tokenizer = Tokenizer(inputCol="name", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

    # Chain transformations
    df_tokenized = tokenizer.transform(df)
    df_filtered = stopwords_remover.transform(df_tokenized)
    logger.info("Tokenization and stopword removal completed")

    # Collect filtered words for embedding
    filtered_words = df_filtered.select("filtered_words").collect()
    documents = [" ".join(row["filtered_words"]) for row in filtered_words]

    # Initialize fastembed TextEmbedding
    logger.info("Generating embeddings with fastembed...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embeddings = list(embedding_model.embed(documents))
    embeddings = [emb.astype(np.float32).tolist()[:128] for emb in embeddings]  # Truncate to 128 dimensions

    # Define schema for embeddings
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("vector", ArrayType(FloatType()), False)
    ])

    # Add embeddings to DataFrame
    df_filtered = df_filtered.withColumn("id", F.monotonically_increasing_id().cast("string"))
    embeddings_rdd = spark.sparkContext.parallelize([(str(i), emb) for i, emb in enumerate(embeddings)])
    embeddings_df = spark.createDataFrame(embeddings_rdd, schema=schema)
    df_vectorized = df_filtered.join(embeddings_df, "id").drop("id")

    # Format event_time and select final columns
    df_final = df_vectorized.withColumn(
        "event_time",
        concat_ws(" ", date_format(col("event_time"), "yyyy-MM-dd HH:mm:ss"), lit("UTC"))
    ).select(
        "user_id",
        "product_id",
        "name",
        "vector",
        "brand",
        "category_code",
        "price",
        "event_type",
        "event_time",
        "user_session"
    )

    # Log schema for debugging
    logger.info("Schema of df_final:")
    df_final.printSchema()

    # Qdrant write options
    qdrant_grpc_url = os.getenv("QDRANT_GRPC_URL")
    collection_name = os.getenv("QDRANT_TEST_COLLECTION")
    options = {
        "qdrant_url": qdrant_grpc_url,
        "collection_name": collection_name,
        "embedding_field": "vector",
        "schema": df_final.schema.json()
    }

    # Write to Qdrant
    logger.info(f"Writing data to Qdrant collection: {collection_name}...")
    df_final.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .option("batch.size", 10000) \
        .save()
    
    logger.info("✅ Data successfully inserted into Qdrant.")

except Exception as e:
    logger.error(f"Error in processing: {str(e)}")
    raise

finally:
    spark.stop()
    logger.info("Spark session stopped")