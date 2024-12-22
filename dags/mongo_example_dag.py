from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import os
from pymongo import MongoClient

def connect_to_mongodb():
    """
    Connects to MongoDB using the MONGODB_URI environment variable,
    inserts a sample document, and prints the inserted document ID.
    """
    mongodb_uri = os.getenv('MONGODB_URI')
    
    if not mongodb_uri:
        raise ValueError("MONGODB_URI environment variable not set")

    try:
        client = MongoClient(mongodb_uri)
        db = client.get_default_database()  # Now, this will return 'mydatabase'
        collection = db['mycollection']
        
        # Example Operation: Insert a Document
        document = {"name": "Bob", "age": 25}
        result = collection.insert_one(document)
        print(f"Inserted document ID: {result.inserted_id}")
    except Exception as e:
        print(f"An error occurred while connecting to MongoDB: {e}")
        raise

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 12, 20),  # Set to a past date
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'mongo_example_dag',
    default_args=default_args,
    description='A simple DAG to connect to MongoDB and perform operations',
    schedule_interval='@daily',
    catchup=False,  # Prevents backfilling
) as dag:
    
    # Start Dummy Task
    start_job = DummyOperator(
        task_id='start_job'
    )
    
    # Task to Connect to MongoDB
    task_connect = PythonOperator(
        task_id='connect_to_mongodb',
        python_callable=connect_to_mongodb
    )
    
    # End Dummy Task
    end_job = DummyOperator(
        task_id='end_job'
    )
    
    # Define Task Dependencies
    start_job >> task_connect >> end_job
