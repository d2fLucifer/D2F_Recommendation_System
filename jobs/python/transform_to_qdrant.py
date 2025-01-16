#!/usr/bin/env python3
import sys
import uuid
import logging

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client import models

# PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_timestamp, when, sum as spark_sum,
    lag, unix_timestamp, col, lit
)
from pyspark.sql.window import Window

# Spark ML libraries
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.functions import vector_to_array

# Custom Spark session creation
from spark_session import create_spark_session

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize Spark session
    spark = create_spark_session("Transform to Qdrant")

    # MongoDB connection settings
    mongo_uri = "mongodb://root:example@mongo:27017/admin"
    db_name = "recommendation_system"
    collection_name = "user_behavior"

    try:
        # Load data from MongoDB
        df = (
            spark.read.format("mongo")
            .option("uri", mongo_uri)
            .option("database", db_name)
            .option("collection", collection_name)
            .load()
        )
    except Exception as e:
        logger.error(f"Failed to load data from MongoDB: {e}")
        spark.stop()
        sys.exit(1)

    logger.info("Data loaded from MongoDB:")
    df.show(10)

    # (Optional) Repartition for parallelism
    df = df.repartition(200)

    # ------------------------------------------------------------------------
    # 4. Convert `event_time` to Timestamp
    # ------------------------------------------------------------------------
    df = df.withColumn(
        "event_time",
        to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
    )

    # ------------------------------------------------------------------------
    # 5. Generate flags for event types
    # ------------------------------------------------------------------------
    df = (
        df.withColumn("view_count", when(col("event_type") == "view", 1).otherwise(0))
          .withColumn("cart_count", when(col("event_type") == "cart", 1).otherwise(0))
          .withColumn("purchase_count", when(col("event_type") == "purchase", 1).otherwise(0))
    )

    # ------------------------------------------------------------------------
    # 6. Compute session-level totals
    # ------------------------------------------------------------------------
    df_totals = df.groupBy("user_session").agg(
        spark_sum("view_count").alias("total_views"),
        spark_sum("cart_count").alias("total_carts"),
        spark_sum("purchase_count").alias("total_purchases")
    )

    # ------------------------------------------------------------------------
    # 7. Compute product-level features (views, carts, purchases) and join
    # ------------------------------------------------------------------------
    df_product = df.groupBy("user_session", "product_id", "name", "user_id").agg(
        spark_sum("view_count").alias("product_views"),
        spark_sum("cart_count").alias("product_carts"),
        spark_sum("purchase_count").alias("product_purchases")
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

    # ------------------------------------------------------------------------
    # 8. Create window for time-based features
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # 9. Aggregate time spent per product vs. total
    # ------------------------------------------------------------------------
    df_time_agg = df_time.groupBy("user_session", "product_id", "user_id").agg(
        spark_sum("time_spent").alias("product_time_spent")
    )

    df_total_time = df_time.groupBy("user_session").agg(
        spark_sum("time_spent").alias("total_time_spent")
    )

    df_features = (
        df_features
        .join(df_time_agg, on=["user_session", "product_id", "user_id"], how="left")
        .join(df_total_time, on="user_session", how="left")
    )

    df_features = df_features.withColumn(
        "F4",
        when(col("total_time_spent") != 0, col("product_time_spent") / col("total_time_spent"))
        .otherwise(0)
    )

    # ------------------------------------------------------------------------
    # 10. Define weights and compute a final score
    # ------------------------------------------------------------------------
    w1, w2, w3, w4 = 0.1, 0.25, 0.45, 0.2
    df_features = (
        df_features
        .withColumn(
            "score",
            w1 * col("F1") + w2 * col("F2") + w3 * col("F3") + w4 * col("F4")
        )
        .fillna({"score": 0})
    )

    # Select relevant columns, including product_name
    final_df = df_features.select("user_id", "product_id", "name", "score")
    logger.info("Feature engineering completed.")
    final_df.show(10)


    spark.stop()
    logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()
