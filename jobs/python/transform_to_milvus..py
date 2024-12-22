# transform_to_milvus.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from spark_session import create_spark_session
import os 
def main():
    spark = create_spark_session("Transform Data to Milvus")
    
    # Read the transformed CSV from MinIO
    df = spark.read.csv("s3a://recommendation/transformed/user-behavior.csv", header=True, inferSchema=True)
    
    # Data Transformation
    transformed_df = df.select(
        to_timestamp(col("event_time")).alias("event_time"),
        col("event_type"),
        col("product_id"),
        col("category_id"),
        col("category_code"),
        col("brand"),
        col("price").cast("double"),
        col("user_id"),
        col("user_session")
    )
    
    # Show transformed data (optional for debugging)
    transformed_df.show(5)
    
    # Collect data to driver for Milvus ingestion (for demonstration; consider scalability)
    data = transformed_df.collect()
    
    # Connect to Milvus
    connections.connect (
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530"),
        alias= "default"
    )
    
    # Define Milvus collection schema
    fields = [
        FieldSchema(name="event_time", dtype=DataType.TIMESTAMP, description="Event Time"),
        FieldSchema(name="event_type", dtype=DataType.VARCHAR, max_length=255, description="Event Type"),
        FieldSchema(name="product_id", dtype=DataType.INT64, description="Product ID"),
        FieldSchema(name="category_id", dtype=DataType.INT64, description="Category ID"),
        FieldSchema(name="category_code", dtype=DataType.VARCHAR, max_length=255, description="Category Code"),
        FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=255, description="Brand"),
        FieldSchema(name="price", dtype=DataType.DOUBLE, description="Price"),
        FieldSchema(name="user_id", dtype=DataType.INT64, description="User ID"),
        FieldSchema(name="user_session", dtype=DataType.VARCHAR, max_length=255, description="User Session"),
    ]
    
    schema = CollectionSchema(fields, description="Recommendation Data")
    
    # Create or load collection
    collection_name = "recommendation_collection"
    if collection_name not in connections.list_collections():
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)
    
    # Prepare data for insertion
    insert_data = [
        [row.event_time for row in data],
        [row.event_type for row in data],
        [row.product_id for row in data],
        [row.category_id for row in data],
        [row.category_code for row in data],
        [row.brand for row in data],
        [row.price for row in data],
        [row.user_id for row in data],
        [row.user_session for row in data],
    ]
    
    # Insert data into Milvus
    collection.insert(insert_data)
    
    spark.stop()

if __name__ == "__main__":
    main()
