# Step 1: Set Up the Environment
# Install necessary packages
# !pip install pyspark qdrant-client sentence-transformers

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Initialize Spark session with Qdrant-Spark connector
spark = SparkSession.builder \
    .config("spark.jars", "/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.instances", "1") \
    .config("spark.executor.cores", "1") \
    .master("local[*]") \
    .appName("QdrantIntegration") \
    .getOrCreate()

# Step 2: Initialize Qdrant Client
qdrant_host = "qdrant"  # Replace with your Qdrant host
qdrant_grpc_port = 6334  # gRPC port
collection_name = "test_collection"

# Initialize Qdrant Client using gRPC
qdrant_client = QdrantClient(host=qdrant_host, port=6333, prefer_grpc=True)

# Define vector parameters
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(embedding_model_name)
vector_size = model.get_sentence_embedding_dimension()

# Create the collection in Qdrant
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
)

# Step 3: Prepare Data
# Sample data
data = [("Alice",), ("Bob",), ("Charlie",)]
df = spark.createDataFrame(data, ["name"])

# Define a UDF to generate embeddings
def generate_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()

embedding_udf = udf(generate_embedding, ArrayType(FloatType()))

# Apply the UDF to create a new column with embeddings
df_with_embeddings = df.withColumn("embedding", embedding_udf(df["name"]))

# Step 4: Insert Data into Qdrant
# Define options for the Qdrant-Spark connector
options = {
    "qdrant_url": f"http://{qdrant_host}:{qdrant_grpc_port}",
    "collection_name": collection_name,
    "embedding_field": "embedding",
    "schema": df_with_embeddings.schema.json()
}

# Write DataFrame to Qdrant without the 'id' column
df_with_embeddings.write \
    .format("io.qdrant.spark.Qdrant") \
    .options(**options) \
    .mode("append") \
    .save()

# Step 5: Verify Data Insertion
# Retrieve points from Qdrant
retrieved_points, _ = qdrant_client.scroll(collection_name=collection_name, limit=10)

# Display retrieved points
for point in retrieved_points:
    print(f"ID: {point.id}, Payload: {point.payload}, Vector: {point.vector}")
