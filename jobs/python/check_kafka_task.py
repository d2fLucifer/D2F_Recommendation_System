from kafka import KafkaConsumer, TopicPartition

# Configuration
KAFKA_BROKER = "kafka.d2f.io.vn:9092"
TOPIC = "model_retrain_event"

def count_messages(bootstrap_servers, topic, group_id=None):
    """
    Calculate the total number of messages in a Kafka topic.
    
    Args:
        bootstrap_servers (str): Kafka bootstrap servers
        topic (str): Name of the Kafka topic
        group_id (str, optional): Consumer group ID
    
    Returns:
        int: Total number of messages in the topic
    """
    try:
        # Initialize Kafka consumer
        consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            group_id=group_id
        )

        # Get topic partitions
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            raise ValueError(f"Topic '{topic}' does not exist or has no partitions")

        total_messages = 0
        
        # Calculate messages for each partition
        for partition in partitions:
            tp = TopicPartition(topic, partition)
            
            # Assign the partition to consumer
            consumer.assign([tp])
            
            # Get earliest offset
            consumer.seek_to_beginning(tp)
            earliest_offset = consumer.position(tp)
            
            # Get latest offset
            consumer.seek_to_end(tp)
            latest_offset = consumer.position(tp)
            
            # Calculate message count for this partition
            partition_messages = latest_offset - earliest_offset
            total_messages += partition_messages
            
            print(f"Partition {partition}: {partition_messages} messages "
                  f"(offsets {earliest_offset} to {latest_offset})")

        consumer.close()
        return total_messages

    except Exception as e:
        print(f"Error: {str(e)}")
        return -1

def main():
    print(f"\nCalculating total messages in topic '{TOPIC}'...")
    print(f"Connecting to Kafka broker: {KAFKA_BROKER}")
    
    total = count_messages(
        bootstrap_servers=KAFKA_BROKER,
        topic=TOPIC,
        group_id=None  # You can specify a group ID here if needed
    )

    if total >= 0:
        print(f"\nTotal messages in topic '{TOPIC}': {total}")
    else:
        print("Failed to calculate message count")

if __name__ == "__main__":
    main()