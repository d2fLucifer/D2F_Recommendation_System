#!/usr/bin/env python3
import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_timestamp, when, sum as spark_sum,
    lag, unix_timestamp, col, lit, udf
)
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.sql.types import ArrayType, FloatType
from spark_session import create_spark_session
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pyspark.sql.functions import col
import uuid
# Logging Setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize Spark
    spark = create_spark_session()

    # MongoDB connection settings
    mongo_uri = "mongodb://root:example@mongo:27017/admin"
    db_name = "recommendation_system"
    collection_name = "user_behavior"

    # Load Data from MongoDB
    try:
        df = (
            spark.read.format("mongo")
            .option("uri", mongo_uri)
            .option("database", db_name)
            .option("collection", collection_name)
            .option("inferSchema", "true")  # Auto detect schema
            .load()
        )

        if df.isEmpty():
            logger.warning("MongoDB collection is empty!")
            spark.stop()
            return
    except Exception as e:
        logger.error(f"Error loading MongoDB data: {e}")
        spark.stop()
        return

    logger.info("Data loaded from MongoDB:")
    df.show(10)
    df = df.limit(10)

    # Feature Engineering
    df = df.withColumn("event_time", to_timestamp("event_time", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
    df = df.withColumn("view_count", when(col("event_type") == "view", 1).otherwise(0))
    df = df.withColumn("cart_count", when(col("event_type") == "cart", 1).otherwise(0))
    df = df.withColumn("purchase_count", when(col("event_type") == "purchase", 1).otherwise(0))

    # Aggregates per session
    df_totals = df.groupBy("user_session").agg(
        spark_sum("view_count").alias("total_views"),
        spark_sum("cart_count").alias("total_carts"),
        spark_sum("purchase_count").alias("total_purchases")
    )

    df_product = df.groupBy("user_session", "product_id", "user_id", "name").agg(
        spark_sum("view_count").alias("product_views"),
        spark_sum("cart_count").alias("product_carts"),
        spark_sum("purchase_count").alias("product_purchases")
    )

    df_features = df_product.join(df_totals, on="user_session", how="left").fillna(0)

    df_features = df_features.withColumn("F1", when(col("total_views") > 0, col("product_views") / col("total_views")).otherwise(0))
    df_features = df_features.withColumn("F2", when(col("total_carts") > 0, col("product_carts") / col("total_carts")).otherwise(0))
    df_features = df_features.withColumn("F3", when(col("total_purchases") > 0, col("product_purchases") / col("total_purchases")).otherwise(0))

    # Time Spent Feature
    window_order = Window.partitionBy("user_session").orderBy("event_time")
    df_time = df.withColumn("prev_event_time", lag("event_time").over(window_order))
    df_time = df_time.withColumn("time_spent", (unix_timestamp("event_time") - unix_timestamp("prev_event_time")).cast("double"))
    df_time = df_time.na.fill({"time_spent": 0})

    df_time_agg = df_time.groupBy("user_session", "product_id", "user_id").agg(
        spark_sum("time_spent").alias("product_time_spent")
    )

    df_total_time = df_time.groupBy("user_session").agg(
        spark_sum("time_spent").alias("total_time_spent")
    )

    df_features = df_features.join(df_time_agg, on=["user_session", "product_id", "user_id"], how="left")
    df_features = df_features.join(df_total_time, on="user_session", how="left").fillna(0)

    df_features = df_features.withColumn("F4", when(col("total_time_spent") > 0, col("product_time_spent") / col("total_time_spent")).otherwise(0))

    # Weighted Score Calculation
    w1, w2, w3, w4 = 0.1, 0.25, 0.45, 0.2
    df_features = df_features.withColumn(
        "score",
        (w1 * col("F1") + w2 * col("F2") + w3 * col("F3") + w4 * col("F4")).cast("double")
    ).fillna(0)

    # Final DataFrame
    final_df = df_features.select("user_id", "product_id", "name", "score")
    logger.info("Final DataFrame:")
    final_df.show(10)

    # NLP Feature Engineering
    tokenizer = Tokenizer(inputCol="name", outputCol="words")
    df_tokenized = tokenizer.transform(df)

    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_filtered = stopwords_remover.transform(df_tokenized)

    word2Vec = Word2Vec(vectorSize=128, minCount=0, inputCol="filtered_words", outputCol="word2vec_features")
    model = word2Vec.fit(df_filtered)
    df_vectorized = model.transform(df_filtered)

    to_list_udf = udf(lambda vec: vec.toArray().tolist(), ArrayType(FloatType()))
    df_final = df_vectorized.withColumn("vector", to_list_udf(col("word2vec_features")))

    df_final = df_final.select("user_id", "product_id", "name", "vector")
    logger.info("Vectorized DataFrame:")
    df_final.show(10, truncate=False)
    QDRANT_HOST = "qdrant"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "user_behaviour"

    # Initialize Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 128, "distance": "Cosine"},  # Adjust size according to word2vec vector size
    )

    
    ## Step 5: Configure connection options for Qdrant
    options = {
        "qdrant_url": "http://qdrant:6334",  # Ensure this is the correct gRPC URL
        "collection_name": COLLECTION_NAME,
        "schema": df_final.schema.json(),
        "embedding_field": "vector", 
    }

    # Step 6: Write DataFrame to Qdrant
    try:
        # Debugging: Print schema and data
        df_final.printSchema()
        df_final.show(truncate=False)

        df_final.write.format("io.qdrant.spark.Qdrant") \
            .options(**options) \
            .mode("append") \
            .save()
        print("✅ Data successfully inserted into Qdrant.")
    except Exception as e:
        print(f"❌ Error inserting data into Qdrant: {e}")

        # Stop Spark
    spark.stop()
    logger.info("Spark session stopped.")

if __name__ == "__main__":
    main()
