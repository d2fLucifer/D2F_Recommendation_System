# spark_submit.py

# Import statements
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, udf, date_format, lit
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from qdrant_client import QdrantClient
from spark_session import create_spark_session
import logging
from pyspark import StorageLevel

# Constants
KAFKA_BROKER = "kafka:9092"
TOPIC = "user-behavior-events"
MONGO_URI = "mongodb://root:example@103.155.161.100:27017/recommendation_system?authSource=admin"
QDRANT_URL = "http://103.155.161.100:6333"
QDRANT_GPRC_URL = "http://103.155.161.100:6334"
COLLECTION_NAME = "recommendation_system111"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark with optimized settings
spark = create_spark_session(app_name="QdrantInsert")

try:
    # Read MongoDB data - User Behaviors with selective columns
    df_userbehaviors = spark.read.format("mongo") \
        .option("uri", MONGO_URI) \
        .option("database", "recommendation_system") \
        .option("collection", "userbehaviors") \
        .option("partitioner", "MongoSamplePartitioner") \
        .load() \
        .select("user_id", "product_id", "event_type", "event_time", "user_session") \
        .repartition(32)  # Adjusted for local[*] with 8 cores
    logger.info(f"Read data from MongoDB 'userbehaviors'. Row count: {df_userbehaviors.count()}")

    # Persist with spill to disk
    df_userbehaviors.persist(StorageLevel.MEMORY_AND_DISK)

    # Read MongoDB data - Products with selective columns
    df_products = spark.read.format("mongo") \
        .option("uri", MONGO_URI) \
        .option("database", "recommendation_system") \
        .option("collection", "products") \
        .option("partitioner", "MongoSamplePartitioner") \
        .load() \
        .select("product_id", "name", "brand", "category", "type", "price")
    logger.info(f"Read data from MongoDB 'products'. Row count: {df_products.count()}")

    # Create category_code column efficiently
    df_products = df_products.withColumn(
        "category_code", 
        concat_ws(".", col("category"), col("type"))
    ).drop("category", "type").persist(StorageLevel.MEMORY_AND_DISK)

    # Join without broadcast (let Spark optimize)
    df_join = df_userbehaviors.join(
        df_products,
        df_userbehaviors.product_id == df_products.product_id,
        "inner"
    ).select(
        "user_id",
        df_userbehaviors.product_id.alias("product_id"),
        "event_type",
        "event_time",
        "user_session",
        col("name").alias("product_name"),
        "brand",
        "category_code",
        "price"
    )

    # Unpersist intermediate DataFrames
    df_userbehaviors.unpersist()
    df_products.unpersist()

    # NLP Feature Engineering with optimized settings
    tokenizer = Tokenizer(inputCol="product_name", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    word2vec = Word2Vec(
        vectorSize=128,  # Reduced from 64
        minCount=10,    # Increased from 5
        inputCol="filtered_words",
        outputCol="word2vec_features",
        numPartitions=32
    )

    # Chain transformations and persist after Word2Vec
    df_vectorized = word2vec.fit(
        stopwords_remover.transform(
            tokenizer.transform(df_join)
        )
    ).transform(
        stopwords_remover.transform(
            tokenizer.transform(df_join)
        )
    ).persist(StorageLevel.MEMORY_AND_DISK)
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
            col("product_name").alias("name"),
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

    # Initialize Qdrant client and recreate collection
    client = QdrantClient(url=QDRANT_URL)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 128, "distance": "Cosine"}  # Match vectorSize
    )
    logger.info("Qdrant collection created successfully")

    # Write to Qdrant with batching
    options = {
        "qdrant_url": QDRANT_GPRC_URL,
        "collection_name": COLLECTION_NAME,
        "embedding_field": "vector",
        "schema": df_final.schema.json()
    }

    df_final.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .option("batch.size", 10000) \
        .save()
    logger.info("✅ Data successfully inserted into Qdrant.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    # Cleanup and stop Spark session
    spark.catalog.clearCache()
    spark.stop()
    logger.info("Spark session stopped.")