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
from pyspark.ml.linalg import DenseVector, SparseVector
from spark_session import create_spark_session          
# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)



# ------------------------------------------------------------------------------
# 2. Main Function
# ------------------------------------------------------------------------------
def main():
    # --------------------------------------------------------------------------
    # 2.1 Initialize Spark
    # --------------------------------------------------------------------------
    spark = create_spark_session("Transform to Qdrant")

    # MongoDB connection settings
    mongo_uri = "mongodb://root:example@mongo:27017/admin"
    db_name = "recommendation_system"
    collection_name = "user_behavior"

    # --------------------------------------------------------------------------
    # 2.2 Load Data from MongoDB
    # --------------------------------------------------------------------------
    df = (
        spark.read.format("mongo")
        .option("uri", mongo_uri)
        .option("database", db_name)
        .option("collection", collection_name)
        .load()
    )

    logger.info("Data loaded from MongoDB:")
    df.show(10)

    # Repartition for potential performance benefits
    df = df.repartition(200)

    # Parse event_time as timestamp
    df = df.withColumn("event_time", to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))

    # --------------------------------------------------------------------------
    # 2.3 Feature Engineering
    # --------------------------------------------------------------------------
    # Basic counts
    df = (
        df.withColumn("view_count", when(col("event_type") == "view", 1).otherwise(0))
          .withColumn("cart_count", when(col("event_type") == "cart", 1).otherwise(0))
          .withColumn("purchase_count", when(col("event_type") == "purchase", 1).otherwise(0))
    )

    # Aggregates per session
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

    # --------------------------------------------------------------------------
    # 2.4 Time Spent Feature
    # --------------------------------------------------------------------------
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
            when(col("total_time_spent") != 0,
                 col("product_time_spent") / col("total_time_spent")).otherwise(0)
        )
    )

    # --------------------------------------------------------------------------
    # 2.5 Weighted Score
    # --------------------------------------------------------------------------
    w1, w2, w3, w4 = 0.1, 0.25, 0.45, 0.2
    df_features = df_features.withColumn(
        "score",
        w1 * col("F1") + w2 * col("F2") + w3 * col("F3") + w4 * col("F4")
    ).fillna({"score": 0})

    # --------------------------------------------------------------------------
    # 2.6 TF-IDF Pipeline on 'name'
    # --------------------------------------------------------------------------
    tokenizer = Tokenizer(inputCol="name", outputCol="words")
    df_tokens = tokenizer.transform(df_features)

    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_filtered = remover.transform(df_tokens)

    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=1000)
    df_hashed = hashing_tf.transform(df_filtered)

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model = idf.fit(df_hashed)
    df_tfidf = idf_model.transform(df_hashed)

    # --------------------------------------------------------------------------
    # 2.7 Build Final DataFrame
    # --------------------------------------------------------------------------
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

    logger.info("Final DataFrame sample:")
    final_df.show(10, truncate=False)

    # ======================================================================
    # Qdrant Integration (Distributed approach, no toPandas())
    # ======================================================================
    QDRANT_HOST = "qdrant"  # Replace if needed
    QDRANT_PORT = 6333      # Default Qdrant port
    QDRANT_COLLECTION_NAME = "recommendation_collection"

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create (or recreate) the collection in Qdrant
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=1000, distance=models.Distance.COSINE),
    )
    logger.info("Qdrant collection created or recreated.")

  
    



    # Prepare data for Qdrant
    payload = [
        {
            "id": str(uuid.uuid4()),
            "vector": row['vector'].toArray().tolist() if isinstance(row['vector'], DenseVector) else row['vector'].toArray().tolist(),
            "payload": {
                "user_id": row['user_id'],
                "product_id": row['product_id'],
                "name": row['name'],
                "score": row['score']
            }
        }
        for row in sample_data
    ]

    # Insert into Qdrant
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=payload
    )

    # logger.info("1000 rows inserted into Qdrant successfully.")



    # ----------------------------------------------------------------------
    # 2.9 Stop Spark
    # ----------------------------------------------------------------------
    spark.stop()
    logger.info("Spark session stopped.")

# ------------------------------------------------------------------------------
# 3. Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
