#!/bin/bash

set -e

echo "Waiting for Kafka to be ready..."
while ! nc -z kafka 29092; do
    sleep 1
done
echo "Kafka is ready!"

echo "Waiting for Qdrant to be ready..."
while ! nc -z qdrant 6333; do
    sleep 1
done
echo "Qdrant is ready!"

echo "Starting Kafka consumer..."
exec python kafka_consumer.py