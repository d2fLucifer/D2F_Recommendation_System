from kafka import KafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment variables
QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'user_events')
COLLECTION_NAME = os.getenv('QDRANT_TEST_COLLECTION', 'test_v2')
KAFKA_GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'user_events_group')

# Initialize Qdrant client with specific configuration
qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    prefer_grpc=True  # Use gRPC for better performance
)

# Initialize FastEmbed for text embedding
embedding_model = TextEmbedding()

# Create collection in Qdrant if it doesn't exist
VECTOR_SIZE = 384  # FastEmbed default vector size

try:
    # Try to get the collection first
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} already exists")
except Exception as e:
    if f"Collection `{COLLECTION_NAME}` already exists" in str(e):
        print(f"Collection {COLLECTION_NAME} already exists")
    else:
        print(f"Creating collection {COLLECTION_NAME}")
        try:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    max_optimization_threads=4
                )
            )
        except Exception as create_error:
            if f"Collection `{COLLECTION_NAME}` already exists" not in str(create_error):
                print(f"Error creating collection: {create_error}")
                raise

def vectorize_text(text: str) -> list:
    """Convert text to vector using FastEmbed"""
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()

def process_message(message):
    """Process a single Kafka message and store it in Qdrant"""
    try:
        # Parse the message value
        data = json.loads(message.value.decode('utf-8'))
        
        # Create text for embedding (combine relevant fields)
        text_to_embed = f"{data.get('name', '')} {data.get('event_type', '')}"
        
        # Generate vector
        vector = vectorize_text(text_to_embed)
        
        # Create payload
        payload = {
            "user_session": data.get('user_session'),
            "user_id": data.get('user_id'),
            "product_id": data.get('product_id'),
            "name": data.get('name'),
            "event_type": data.get('event_type'),
            "event_time": data.get('event_time')
        }
        
        # Create point
        point = models.PointStruct(
            id=int(time.time() * 1000),  # Generate unique ID based on timestamp
            vector=vector,
            payload=payload
        )
        
        # Upload point to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        print(f"Processed message: {data}")
    except Exception as e:
        print(f"Error processing message: {e}")

def main():
    print("Starting Kafka consumer...")
    print(f"Configuration:")
    print(f"- Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"- Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"- Topic: {KAFKA_TOPIC}")
    print(f"- Collection: {COLLECTION_NAME}")
    print(f"- Group ID: {KAFKA_GROUP_ID}")
    
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id=KAFKA_GROUP_ID
    )
    
    print("Kafka consumer initialized. Waiting for messages...")
    
    # Process messages
    for message in consumer:
        process_message(message)

if __name__ == "__main__":
    main()
