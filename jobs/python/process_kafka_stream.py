from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from spark_session import create_spark_session
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
KAFKA_HOST = "kafka:29092"
TOPIC = "user-behavior-events"
MONGO_URI = "mongodb://root:example@103.155.161.100:27017/recommendation_system?authSource=admin"

# Create Spark session
spark = create_spark_session(app_name="ProcessKafkaBatch")

# Set S3A commit configurations
spark.conf.set("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
spark.conf.set("spark.sql.parquet.output.committer.class", "org.apache.spark.sql.execution.datasources.DirectParquetOutputCommitter")
spark.conf.set("spark.hadoop.fs.s3a.attempts.maximum", "10")

# Define schema
schema = StructType([
    StructField("user_session", StringType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("product_id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("event_time", TimestampType(), True)
])

# Read from Kafka
df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_HOST) \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse Kafka data
parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*").repartition(50, "user_id")
df_userbehaviors = parsed_df

# Write to MongoDB
logger.info("Writing user behaviors to MongoDB...")
df_userbehaviors.write \
    .format("mongo") \
    .option("uri", MONGO_URI) \
    .option("database", "recommendation_system") \
    .option("collection", "userbehaviors") \
    .option("partitioner", "MongoSamplePartitioner") \
    .mode("append") \
    .save()

# Read products from MongoDB
df_products = spark.read.format("mongo") \
    .option("uri", MONGO_URI) \
    .option("database", "recommendation_system") \
    .option("collection", "products") \
    .option("partitioner", "MongoSamplePartitioner") \
    .option("partitionKey", "product_id") \
    .load() \
    .select("product_id", "name", "brand", "category", "type", "price") \
    .repartition(50, "product_id")

# Create category_code
df_products = df_products.withColumn("category_code", concat_ws(".", col("category"), col("type"))).drop("category", "type")

# Join data
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

# Write to S3
logger.info("Writing final data to S3 in CSV format...")
df_final.write.mode("overwrite").csv("s3a://dataset/pretrain_data/", header=True)

# Stop Spark session
logger.info("Processing complete. Stopping Spark session...")
spark.stop()