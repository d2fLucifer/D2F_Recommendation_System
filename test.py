from qdrant_client import QdrantClient

try:
    client = QdrantClient(url="http://qdrant:6333", grpc_port=6334, prefer_grpc=True)
    print("Connected to Qdrant.")
    print(client.get_collections())
except Exception as e:
    print(f"Error: {e}")
