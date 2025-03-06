# Import 
from kafka import KafkaConsumer
import json
from spark_session import create_spark_session

# Create a spark session
spark = create_spark_session()

# Extract data from kafka topic 






# Preprocess data by using mongoDB to match the data with the pre-trained model





# Save the preprocessed data into qdrant database




# Notify AI module that the data is ready for training by sending a message to the kafka topic



# Path: jobs/python/inject_to_mongo_db.py


