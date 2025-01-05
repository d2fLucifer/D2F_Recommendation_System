from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import requests

# Configuration
qdrant_host = "qdrant"  # Replace with your Qdrant host if different
qdrant_http_port = 6333  # HTTP port
collection_name = "test_collection"

# Initialize Qdrant Client using HTTP
client = QdrantClient(host=qdrant_host, port=qdrant_http_port, prefer_grpc=False)

# 1. Test Qdrant Connectivity using HTTP
try:
    response = requests.get(f"http://{qdrant_host}:{qdrant_http_port}/collections")
    if response.status_code == 200:
        print("✅ Successfully connected to Qdrant (HTTP).")
    else:
        print(f"❌ Failed to connect to Qdrant (HTTP): {response.text}")
        exit(1)
except Exception as e:
    print(f"❌ Error connecting to Qdrant (HTTP): {e}")
    exit(1)

# 2. Define Vector Configuration
# Determine the size based on the embedding model you choose
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(embedding_model_name)
vector_size = model.get_sentence_embedding_dimension()

vectors_config = VectorParams(size=vector_size, distance=Distance.COSINE)

# 3. Create or Recreate Collection
try:
    # This will delete the collection if it exists and create a new one
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=vectors_config
    )
    print(f"✅ Collection '{collection_name}' is ready.")
except Exception as e:
    print(f"❌ Failed to create collection '{collection_name}': {e}")
    exit(1)

# 4. Prepare Data Points
names = ["Alice", "Bob", "Charlie"]

# Generate embeddings for each name
embeddings = model.encode(names)

# Create data points with embeddings
data = []
for name, vector in zip(names, embeddings):
    point = {
        "id": str(uuid.uuid4()),
        "payload": {"name": name},
        "vector": vector.tolist()  # Convert numpy array to list
    }
    print("vector", vector)
    data.append(point)

# Convert data to Qdrant PointStruct format
points = [
    PointStruct(id=point["id"], vector=point["vector"], payload=point["payload"])
    for point in data
]

# 5. Insert Points into Qdrant
try:
    client.upsert(collection_name=collection_name, points=points)
    print("✅ Points successfully inserted into Qdrant.")
except Exception as e:
    print(f"❌ Error inserting points into Qdrant: {e}")
    exit(1)

# 6. Verify Data Insertion
try:
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"🔍 Collection info: {collection_info}")

    # Retrieve all points (assuming small dataset; for larger datasets, use pagination)
    retrieved_points, _ = client.scroll(collection_name=collection_name, limit=10)

    if retrieved_points:
        print("✅ Retrieved Points:")
        for point in retrieved_points:
            print(f"ID: {point.id}, Payload: {point.payload}, Vector: {point.vector}")
    else:
        print("⚠️ No points retrieved from the collection.")
except Exception as e:
    print(f"❌ Error retrieving data from Qdrant: {e}")
    exit(1)
