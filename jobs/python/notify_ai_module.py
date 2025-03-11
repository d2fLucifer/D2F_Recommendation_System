from kafka import KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
import json
import sys
from datetime import datetime

KAFKA_BROKER = "kafka.d2f.io.vn:9092"  
TOPIC = "model_retrain_event"

def create_topic_if_not_exists():
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
        existing_topics = admin_client.list_topics()

        if TOPIC not in existing_topics:
            topic = NewTopic(name=TOPIC, num_partitions=1, replication_factor=1)
            admin_client.create_topics([topic])
            print(f"Topic '{TOPIC}' created successfully.")
        else:
            print(f"Topic '{TOPIC}' already exists.")

    except Exception as e:
        print(f"Error checking/creating topic: {e}")

def notify_ai_module():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    message = {
        "event": "data_ready",
        "timestamp": int(datetime.utcnow().timestamp()),
        "message": "New data has been injected to Qdrant. AI module should retrain the model.",
    }

    producer.send(TOPIC, value=message)
    producer.flush()
    print(f"Sent notification to AI module: {message}")

if __name__ == "__main__":
    try:
        create_topic_if_not_exists()
        notify_ai_module()
    except Exception as e:
        print(f"Failed to notify AI module: {str(e)}")
        sys.exit(1)
