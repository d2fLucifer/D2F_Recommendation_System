from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from fastembed import TextEmbedding

# Initialize the embedding model
embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

# Step 1: Create a SparkSession
spark = SparkSession.builder \
    .config("spark.jars", "/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar") \
    .appName("QdrantIntegration") \
    .getOrCreate()

# Step 2: Define Schema for the DataFrame
schema = StructType([
    StructField("id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("embedding", ArrayType(FloatType()), True),  # Changed to ArrayType
])

# Step 3: Generate data with UUID and compute embeddings
def generate_data(names):
    data = []
    for name in names:
        embedding_generator = embedding_model.embed(name)
        try:
            embedding_array = next(embedding_generator)  # Get the first (and likely only) embedding
        except StopIteration:
            print(f"❌ No embedding returned for {name}. Skipping.")
            continue  # Skip if no embedding is returned

        # Convert embedding to a list of floats
        if hasattr(embedding_array, 'tolist'):
            embedding_list = embedding_array.tolist()
        else:
            embedding_list = list(embedding_array)

        # Ensure all elements are floats
        embedding_list = [float(x) for x in embedding_list]

        # Debugging: Print types and sizes
        print(f"Generated embedding for {name}: Size = {len(embedding_list)}")
        print(f"Embedding Data Type: {type(embedding_list)}, Element Type: {type(embedding_list[0])}")

        data.append((str(uuid.uuid4()), name, embedding_list))  # Use list directly
    return data

names = ["Alice", "Bob", "Charlie"]
data = generate_data(names)

# Verify that data is not empty
if not data:
    print("❌ No valid data to process. Exiting.")
    spark.stop()
    exit()

# Verify consistent embedding sizes
vector_size = len(data[0][2])  # Length of the first embedding
for record in data:
    if len(record[2]) != vector_size:
        print(f"❌ Inconsistent embedding size for ID: {record[0]}, Name: {record[1]}")
        spark.stop()
        exit()

# Step 4: Create DataFrame with embeddings
df = spark.createDataFrame(data, schema)

# Step 5: Initialize QdrantClient and Create Collection if not exists
qdrant_url = "http://qdrant:6333"  # Use REST endpoint consistently
collection_name = "test_collection"

client = QdrantClient(url=qdrant_url)

try:
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"✅ Collection `{collection_name}` has been successfully created with vector size {vector_size}.")
except Exception as e:
    print(f"❌ Error checking or creating collection: {e}")
    spark.stop()
    exit()

# Step 6: Configure connection options for Qdrant
options = {
    "qdrant_url": "http://qdrant:6334",  # Use REST endpoint
    "collection_name": collection_name,
    "embedding_field": "embedding",
    "schema" : df.schema.json(),
}

# Step 7: Write DataFrame to Qdrant
try:
    # Debugging: Print schema and data
    df.printSchema()
    df.show(truncate=False)

    # Write to Qdrant
    df.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .save()
    print("✅ Data successfully inserted into Qdrant.")
except Exception as e:
    print(f"❌ Error inserting data into Qdrant: {e}")

# Step 8: Stop Spark Session
spark.stop()
