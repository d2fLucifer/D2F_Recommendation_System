#!/bin/bash

# Exit on error
set -e



echo "Starting Kafka consumer..."

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
while ! nc -z kafka 29092; do
  echo "Waiting for Kafka..."
  sleep 2
done

# Wait for Qdrant to be ready
echo "Waiting for Qdrant to be ready..."
while ! nc -z qdrant 6333; do
  echo "Waiting for Qdrant..."
  sleep 2
done

echo "All services are ready. Starting Kafka consumer..."

# Run the Kafka consumer
exec python kafka.py