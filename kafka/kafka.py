from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, lit
from pyspark.sql.types import StructType, StringType, TimestampType
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
import numpy as np
from typing import List, Dict
import json

# Initialize Qdrant client
qdrant_client = QdrantClient(host="qdrant", port=6333)

# Initialize FastEmbed for text embedding
embedding_model = TextEmbedding()

# Create collection in Qdrant if it doesn't exist
COLLECTION_NAME = "user_events"
VECTOR_SIZE = 384  # FastEmbed default vector size

try:
    qdrant_client.get_collection(COLLECTION_NAME)
except Exception:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        )
    )

def vectorize_text(text: str) -> List[float]:
    """Convert text to vector using FastEmbed"""
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()

def write_to_qdrant(batch_df, batch_id):
    """Write batch data to Qdrant"""
    # Convert Spark DataFrame to list of dictionaries
    records = batch_df.collect()
    
    # Prepare points for Qdrant
    points = []
    for idx, record in enumerate(records):
        # Create payload with all fields
        payload = {
            "user_session": record.user_session,
            "user_id": record.user_id,
            "product_id": record.product_id,
            "name": record.name,
            "event_type": record.event_type,
            "event_time": record.event_time.isoformat() if record.event_time else None
        }
        
        # Create text for embedding (combine relevant fields)
        text_to_embed = f"{record.name} {record.event_type}"
        
        # Generate vector
        vector = vectorize_text(text_to_embed)
        
        # Create point
        point = models.PointStruct(
            id=batch_id * 1000 + idx,  # Generate unique ID
            vector=vector,
            payload=payload
        )
        points.append(point)
    
    # Upload points to Qdrant
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

# 1. Create Spark session
spark = SparkSession.builder \
    .appName("UserEventsProcessor") \
    .getOrCreate()

# 2. Define schema of Kafka message
schema = StructType() \
    .add("user_session", StringType()) \
    .add("user_id", StringType()) \
    .add("product_id", StringType()) \
    .add("name", StringType()) \
    .add("event_type", StringType()) \
    .add("event_time", StringType())

# 3. Read stream from Kafka topic `user_events`
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "user_events") \
    .option("startingOffsets", "latest") \
    .load()

# 4. Parse JSON from Kafka "value"
parsed_df = df.selectExpr("CAST(value AS STRING) as json_string") \
    .select(from_json(col("json_string"), schema).alias("data")) \
    .select("data.*")

# 5. Convert event_time to TimestampType
from pyspark.sql.functions import to_timestamp
parsed_df = parsed_df.withColumn("event_time", to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss z"))

# 6. Write to Qdrant
parsed_df.writeStream \
    .foreachBatch(write_to_qdrant) \
    .outputMode("update") \
    .start() \
    .awaitTermination()
