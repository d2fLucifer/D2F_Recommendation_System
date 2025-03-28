# inject_to_qdrant_retrain.py

# Standard Python imports
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, concat_ws, udf, date_format, lit
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, IntegerType, TimestampType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark import StorageLevel
from spark_session import create_spark_session


# Constants
QDRANT_URL = "http://103.155.161.100:6333"
QDRANT_GPRC_URL = "http://103.155.161.100:6334"
COLLECTION_NAME = "recommendation_system"

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Spark session
logger.info("Creating Spark session...")
spark = create_spark_session(app_name="QdrantInsert")

try:
    # Read data from S3
    logger.info("Reading data from S3...")
    df = spark.read.csv("s3a://dataset/pretrain_data/", header=True)
    df.show(5)

    # NLP Feature Engineering with optimized settings
    logger.info("Starting NLP feature engineering...")
    tokenizer = Tokenizer(inputCol="name", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    word2vec = Word2Vec(
        vectorSize=128,
        minCount=1,
        inputCol="filtered_words",
        outputCol="word2vec_features",
        numPartitions=32
    )

    # Chain transformations and persist after Word2Vec
    df_tokenized = tokenizer.transform(df)
    df_filtered = stopwords_remover.transform(df_tokenized)
    df_vectorized = word2vec.fit(df_filtered).transform(df_filtered)
    logger.info("NLP transformations completed")

    # Convert vector to list and format event_time
    to_list_udf = udf(lambda vec: vec.toArray().tolist(), ArrayType(FloatType()))
    df_final = df_vectorized.withColumn("vector", to_list_udf(col("word2vec_features"))) \
        .withColumn(
            "event_time",
            concat_ws(" ", date_format(col("event_time"), "yyyy-MM-dd HH:mm:ss"), lit("UTC"))
        ) \
        .select(
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
    options = {
        "qdrant_url": QDRANT_GPRC_URL,
        "collection_name": COLLECTION_NAME,
        "embedding_field": "vector",
        "schema": df_final.schema.json()
    }

    # Write to Qdrant
    logger.info("Writing data to Qdrant...")
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