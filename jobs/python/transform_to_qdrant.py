#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_timestamp, when, sum as spark_sum, lag, unix_timestamp, col
)
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window

# Spark ML libraries for text processing
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

# Function to create a Spark session
def create_spark_session(app_name: str) -> SparkSession:
    """
    Creates a SparkSession with recommended settings.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")
        .config("spark.mongodb.input.partitioner", "MongoPaginateBySizePartitioner")
        .getOrCreate()
    )
    return spark

# Initialize the Spark session
spark = create_spark_session("Transform to Qdrant")

# MongoDB connection settings
mongo_uri = "mongodb://root:example@mongo:27017/admin"
db_name = "recommendation_system"
collection_name = "user_behavior"

# Load data from MongoDB
df = (
    spark.read.format("mongo")
    .option("uri", mongo_uri)
    .option("database", db_name)
    .option("collection", collection_name)
    .load()
)

df.show(10)

# Repartition for potential performance benefits
df = df.repartition(200)

# Parse event_time as timestamp
df = df.withColumn("event_time", to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))

# Feature engineering
df = (
    df.withColumn("view_count", when(col("event_type") == "view", 1).otherwise(0))
      .withColumn("cart_count", when(col("event_type") == "cart", 1).otherwise(0))
      .withColumn("purchase_count", when(col("event_type") == "purchase", 1).otherwise(0))
)

df_totals = (
    df.groupBy("user_session")
      .agg(
          spark_sum("view_count").alias("total_views"),
          spark_sum("cart_count").alias("total_carts"),
          spark_sum("purchase_count").alias("total_purchases")
      )
)

df_product = (
    df.groupBy("user_session", "product_id", "user_id", "name")
      .agg(
          spark_sum("view_count").alias("product_views"),
          spark_sum("cart_count").alias("product_carts"),
          spark_sum("purchase_count").alias("product_purchases")
      )
)

df_features = df_product.join(df_totals, on="user_session", how="left")

df_features = (
    df_features
    .withColumn(
        "F1",
        when(col("total_views") != 0, col("product_views") / col("total_views")).otherwise(0)
    )
    .withColumn(
        "F2",
        when(col("total_carts") != 0, col("product_carts") / col("total_carts")).otherwise(0)
    )
    .withColumn(
        "F3",
        when(col("total_purchases") != 0, col("product_purchases") / col("total_purchases")).otherwise(0)
    )
)

# Time spent feature (F4)
window_order = Window.partitionBy("user_session").orderBy("event_time")

df_time = (
    df.withColumn("prev_event_time", lag("event_time").over(window_order))
      .withColumn(
          "time_spent_seconds",
          unix_timestamp("event_time") - unix_timestamp("prev_event_time")
      )
      .na.fill(0, subset=["time_spent_seconds"])
      .withColumn("time_spent", col("time_spent_seconds").cast("double"))
)

df_time_agg = (
    df_time.groupBy("user_session", "product_id", "user_id")
           .agg(spark_sum("time_spent").alias("product_time_spent"))
)

df_total_time = (
    df_time.groupBy("user_session")
           .agg(spark_sum("time_spent").alias("total_time_spent"))
)

df_features = (
    df_features
    .join(df_time_agg, on=["user_session", "product_id", "user_id"], how="left")
    .join(df_total_time, on="user_session", how="left")
    .withColumn(
        "F4",
        when(col("total_time_spent") != 0, col("product_time_spent") / col("total_time_spent")).otherwise(0)
    )
)

# Weighted score computation
w1, w2, w3, w4 = 0.1, 0.25, 0.45, 0.2

df_features = df_features.withColumn(
    "score",
    w1 * col("F1") + w2 * col("F2") + w3 * col("F3") + w4 * col("F4")
)

df_features = df_features.fillna({"score": 0})

# Text embeddings with Spark ML (TF-IDF)
# Step 1: Tokenize the 'name' column
tokenizer = Tokenizer(inputCol="name", outputCol="words")
df_tokens = tokenizer.transform(df_features)

# Step 2: Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = remover.transform(df_tokens)

# Step 3: Apply HashingTF to convert text to term frequency vectors
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
df_hashed = hashing_tf.transform(df_filtered)

# Step 4: Compute the IDF and generate TF-IDF features
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(df_hashed)
df_tfidf = idf_model.transform(df_hashed)

# Optional: If you prefer to reduce the dimensionality or use Word2Vec, you can replace the above steps accordingly.

# Build final DataFrame
final_df = (
    df_tfidf
    .select(
        col("user_id"),
        col("product_id"),
        col("name"),
        col("score"),
        col("tfidf_features").alias("vector")
    )
)

final_df.show(10, truncate=False)

# If you need the vector in a specific format (e.g., list of floats), you can convert it using a UDF.
# However, Spark's MLlib vectors are usually compatible with downstream ML tasks.

# Stop the Spark session
spark.stop()
