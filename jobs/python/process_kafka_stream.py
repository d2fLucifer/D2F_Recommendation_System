from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from spark_session import create_spark_session  # Assuming this is your custom module
from kafka.admin import KafkaAdminClient, NewPartitions
from kafka import KafkaConsumer
from kafka.structs import TopicPartition
from kafka.admin import KafkaAdminClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
KAFKA_HOST = "kafka.d2f.io.vn:9092"
TOPIC = "user-behavior-events"
MONGO_URI = "mongodb://root:example@103.155.161.100:27017/recommendation_system?authSource=admin"



# Create Spark session with optimized configurations
spark = create_spark_session(app_name="ProcessKafkaBatch")

# Define schema matching Kafka message format
schema = StructType([
    StructField("user_session", StringType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("product_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("event_time", TimestampType(), True)
])

# Read batch data from Kafka
df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_HOST) \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()
df.show(5)

# Parse Kafka value efficiently
parsed_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*").repartition(50, "user_id")

# Process user behaviors
df_userbehaviors = parsed_df 

# Write user behaviors to MongoDB in batch mode
logger.info("Writing user behaviors to MongoDB...")
df_userbehaviors.write \
    .format("mongo") \
    .option("uri", MONGO_URI) \
    .option("database", "recommendation_system") \
    .option("collection", "userbehaviors") \
    .option("partitioner", "MongoSamplePartitioner") \
    .mode("append") \
    .save()

# Clear Kafka topic after successful MongoDB write
logger.info("Clearing Kafka topic after MongoDB write...")


KAFKA_HOST = "kafka.d2f.io.vn:9092"
TOPIC_NAME = "user-behavior-events"

# try:
#     admin_client = KafkaAdminClient(
#         bootstrap_servers=KAFKA_HOST,
#         client_id='kafka_cleaner'
#     )

#     # Delete topic
#     admin_client.delete_topics([TOPIC_NAME])
#     print(f"Topic {TOPIC_NAME} has been deleted.")

#     admin_client.close()

# except Exception as e:
#     print(f"Error while deleting topic: {e}")

# Read products from MongoDB (static data)
df_products = spark.read.format("mongo") \
    .option("uri", MONGO_URI) \
    .option("database", "recommendation_system") \
    .option("collection", "products") \
    .option("partitioner", "MongoSamplePartitioner") \
    .option("partitionKey", "product_id") \
    .load() \
    .select("product_id", "name", "brand", "category", "type", "price") \
    .repartition(50, "product_id")

logger.info(f"Read data from MongoDB 'products'. Row count: {df_products.count()}")

# Create category_code column efficiently
df_products = df_products.withColumn(
    "category_code", 
    concat_ws(".", col("category"), col("type"))
).drop("category", "type")

# Join user behaviors with static products
df_final = df_userbehaviors.join(
    df_products.hint("broadcast"),
    df_userbehaviors.product_id == df_products.product_id,
    "inner"
).select(
    "user_id",
    df_userbehaviors.product_id.alias("product_id"),
    "event_type",
    "event_time",
    "user_session",
    df_products.name.alias("name"),
    "brand",
    "category_code",
    "price"
)

# Write final data to Parquet in S3 in batch mode
logger.info("Writing final data to S3 in Parquet format...")
df_final.coalesce(1).write.mode("overwrite").csv("s3a://dataset/pretrain_data/", header=True)

# Stop Spark session
logger.info("Processing complete. Stopping Spark session...")
spark.stop()