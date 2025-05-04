from kafka import KafkaConsumer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
import json
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get configuration from environment variables
QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'user_events')
COLLECTION_NAME = os.getenv('QDRANT_TEST_COLLECTION', 'test_v2')
KAFKA_GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'user_events_group')

# Initialize FastEmbed with BAAI/bge-small-en-v1.5
try:
    logger.info("Initializing FastEmbed model...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    logger.info("FastEmbed model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize FastEmbed model: {e}")
    raise

# Initialize Qdrant client
try:
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        prefer_grpc=True
    )
    logger.info("Successfully connected to Qdrant")
except Exception as e:
    logger.error(f"Failed to connect to Qdrant: {e}")
    raise



def vectorize_text(text: str) -> list:
    """Convert text to vector using FastEmbed and truncate to 128 dimensions"""
    try:
        embeddings = list(embedding_model.embed([text]))
        # Truncate to the first 128 dimensions
        vector = embeddings[0][:128].tolist()
        if len(vector) != 128:
            logger.error(f"Truncated vector has incorrect dimension: {len(vector)}")
            raise ValueError("Vector dimension error after truncation")
        logger.info(f"Generated vector with dimension: {len(vector)}")
        return vector
    except Exception as e:
        logger.error(f"Error in vectorization: {e}")
        raise

def process_message(message):
    """Process a single Kafka message and store it in Qdrant"""
    try:
        # Parse the message value
        data = json.loads(message.value.decode('utf-8'))
        logger.info(f"Processing message: {data}")
        
        # Create text for embedding
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
        
        logger.info(f"Successfully processed message: {data}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise

def main():
    logger.info("Starting Kafka consumer...")
    logger.info(f"Configuration:")
    logger.info(f"- Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    logger.info(f"- Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"- Topic: {KAFKA_TOPIC}")
    logger.info(f"- Collection: {COLLECTION_NAME}")
    logger.info(f"- Group ID: {KAFKA_GROUP_ID}")
    
    # Initialize Kafka consumer
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            group_id=KAFKA_GROUP_ID,
            consumer_timeout_ms=None,
            max_poll_records=100,
            max_poll_interval_ms=300000,
            session_timeout_ms=10000,
            heartbeat_interval_ms=3000,
            request_timeout_ms=30000,
            retry_backoff_ms=1000,
            fetch_max_wait_ms=500,
            fetch_min_bytes=1,
            fetch_max_bytes=52428800
        )
        logger.info("Kafka consumer initialized. Waiting for messages...")
    except Exception as e:
        logger.error(f"Failed to initialize Kafka consumer: {e}")
        raise
    
    # Process messages
    try:
        while True:
            try:
                messages = consumer.poll(timeout_ms=1000)
                
                if not messages:
                    continue
                
                for topic_partition, msgs in messages.items():
                    for message in msgs:
                        try:
                            if message is None or message.value is None:
                                logger.warning("Received None message, skipping...")
                                continue
                                
                            process_message(message)
                            consumer.commit()
                        except Exception as e:
                            logger.error(f"Error processing individual message: {e}")
                            continue
                
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                time.sleep(5)
                continue
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        consumer.close()
        logger.info("Kafka consumer closed")

if __name__ == "__main__":
    main()