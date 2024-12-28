from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_timestamp, when, sum as spark_sum, lag, unix_timestamp, col
)
from pyspark.sql.window import Window
from spark_session import create_spark_session

# Initialize Spark Session with MongoDB Connector
mongo_uri = "mongodb://root:example@mongo:27017/admin"  # Update with your MongoDB URI
db_name = "recommendation_system"
collection_name = "user_behavior"

spark = create_spark_session("Transform to Qdrant")

# Load data from MongoDB
df = spark.read.format("mongo") \
    .option("uri", mongo_uri) \
    .option("database", db_name) \
    .option("collection", collection_name) \
    .load()

df.show(10)
df = df.repartition(200)

# Parse event_time as timestamp with explicit format if necessary
# Adjust the format string based on your actual data
df = df.withColumn("event_time", to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))

# Feature Engineering: Initialize count columns
df = df.withColumn("view_count", when(col("event_type") == "view", 1).otherwise(0)) \
       .withColumn("cart_count", when(col("event_type") == "cart", 1).otherwise(0)) \
       .withColumn("purchase_count", when(col("event_type") == "purchase", 1).otherwise(0))

# Compute total aggregates per user_session
df_totals = df.groupBy("user_session") \
              .agg(
                  spark_sum("view_count").alias("total_views"),
                  spark_sum("cart_count").alias("total_carts"),
                  spark_sum("purchase_count").alias("total_purchases")
              )

# Compute product-level aggregates per user_session, product_id, user_id
df_product = df.groupBy("user_session", "product_id", "user_id") \
               .agg(
                   spark_sum("view_count").alias("product_views"),
                   spark_sum("cart_count").alias("product_carts"),
                   spark_sum("purchase_count").alias("product_purchases")
               )

# Join the two DataFrames on user_session
df_features = df_product.join(df_totals, on="user_session", how="left")

# Compute F1, F2, F3 with division by zero handling
df_features = df_features.withColumn(
    "F1",
    when(col("total_views") != 0, col("product_views") / col("total_views")).otherwise(0)
).withColumn(
    "F2",
    when(col("total_carts") != 0, col("product_carts") / col("total_carts")).otherwise(0)
).withColumn(
    "F3",
    when(col("total_purchases") != 0, col("product_purchases") / col("total_purchases")).otherwise(0)
)

# Feature Engineering for F4 (Time Spent)
window_order = Window.partitionBy("user_session").orderBy("event_time")

df_time = df.withColumn("prev_event_time", lag("event_time").over(window_order)) \
           .withColumn(
               "time_spent_seconds",
               unix_timestamp("event_time") - unix_timestamp("prev_event_time")
           ) \
           .na.fill(0, subset=["time_spent_seconds"]) \
           .withColumn("time_spent", col("time_spent_seconds").cast("double"))

# Aggregate time spent per product
df_time_agg = df_time.groupBy("user_session", "product_id", "user_id") \
                     .agg(
                         spark_sum("time_spent").alias("product_time_spent")
                     )

# Aggregate total time spent per user_session
df_total_time = df_time.groupBy("user_session") \
                       .agg(
                           spark_sum("time_spent").alias("total_time_spent")
                       )

# Join time aggregates with df_features
df_features = df_features.join(df_time_agg, on=["user_session", "product_id", "user_id"], how="left") \
                         .join(df_total_time, on="user_session", how="left")

# Compute F4 with division by zero handling
df_features = df_features.withColumn(
    "F4",
    when(col("total_time_spent") != 0, col("product_time_spent") / col("total_time_spent")).otherwise(0)
)

# Define weights
w1 = 0.1
w2 = 0.25
w3 = 0.45
w4 = 0.2

# Compute score
df_features = df_features.withColumn(
    "score",
    w1 * col("F1") + w2 * col("F2") + w3 * col("F3") + w4 * col("F4")
)

# Handle possible nulls in score
df_features = df_features.fillna({'score': 0})

# Select final columns
final_df = df_features.select("user_id", "product_id", "score")

final_df.show(10, truncate=False)

# Stop the Spark session
spark.stop()
