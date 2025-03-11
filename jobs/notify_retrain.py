from datetime import datetime
from kafka import KafkaProducer
import json

producer = KafkaProducer(
        bootstrap_servers="kafka.d2f.io.vn:9092",
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
message = {
        "event": "DATA_READY",
        "timestamp": str(datetime.utcnow()),
        "dataset_version": "v1.0",
    }
producer.send("model_retrain_event", value=message)
producer.flush()
producer.close()