from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Step 1: Create a SparkSession
spark = SparkSession.builder \
    .config("spark.jars", "/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar") \
    .appName("QdrantIntegration") \
    .getOrCreate()

# Step 2: Define Schema for the DataFrame
schema = StructType([
    StructField("id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True)
])

# Step 3: Generate data with UUID
data = [
    (str(uuid.uuid4()), "Alice", [0.1, 0.2, 0.3]),
    (str(uuid.uuid4()), "Bob", [0.4, 0.5, 0.6]),
    (str(uuid.uuid4()), "Charlie", [0.7, 0.8, 0.9])
]

df = spark.createDataFrame(data, schema)

# Step 4: Initialize QdrantClient and Create Collection if not exists
qdrant_url = "http://qdrant:6333"  # Replace if necessary
collection_name = "test_collection"

client = QdrantClient(url=qdrant_url)

try:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    print(f"✅ Collection `{collection_name}` has been successfully created.")
except Exception as e:
    print(f"❌ Error checking or creating collection: {e}")
    spark.stop()
    exit()

# Step 5: Configure connection options for Qdrant
options = {
    "qdrant_url": "http://qdrant:6334",  # Ensure this is the correct gRPC URL
    "collection_name": collection_name,
    "schema": df.schema.json(),
    "embedding_field": "embedding",  # Use 'vector_field' instead of 'vector_fields'
}

# Step 6: Write DataFrame to Qdrant
try:
    # Debugging: Print schema and data
    df.printSchema()
    df.show(truncate=False)

    df.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .save()
    print("✅ Data successfully inserted into Qdrant.")
except Exception as e:
    print(f"❌ Error inserting data into Qdrant: {e}")

# Step 7: Stop Spark Session
spark.stop()