from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_ENDPOINT = "http://host.docker.internal:11434/api/generate"
MONGO_URI = "mongodb://root:example@103.155.161.94:27017/?authSource=admin"
DATABASE_NAME = "recommendation_system"
COLLECTION_NAME = "products"

def create_spark_session():
    """Create and return a Spark session configured for MongoDB."""
    return SparkSession.builder \
        .appName("ProductDescriptionGenerator") \
        .config("spark.mongodb.input.uri", MONGO_URI) \
        .config("spark.mongodb.output.uri", MONGO_URI) \
        .config("spark.mongodb.database", DATABASE_NAME) \
        .config("spark.mongodb.collection", COLLECTION_NAME) \
        .getOrCreate()

def generate_description(name):
    """Generate a product description using the Ollama API based on the product name."""
    if not name:
        return "No product information available."

    prompt = f"Product Name: {name}\nGenerate a product description based on the product name."

    payload = {
        "model": "llama2:7b",
        # "messages": [{"role": "user", "content": prompt}],
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()

        # Sửa lỗi lấy content từ response
        response_content = result.get("response", "{}")
        parsed_response = json.loads(response_content)
        logger.info(f"Generated description: {parsed_response}")
        description = parsed_response.get("message", "Failed to generate description.")

        return description
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        return f"Error generating description: {str(e)}"

def process_dataframe():
    """Load data from MongoDB, generate descriptions, and display the results."""
    spark = create_spark_session()
    
    try:
        # Load data from MongoDB
        df = spark.read.format("mongo").option("database", DATABASE_NAME).option("collection", COLLECTION_NAME).load()
        df = df.limit(1)  # Limit for testing

        # Define UDF for description generation
        description_udf = udf(generate_description, StringType())

        # Apply UDF to DataFrame
        result_df = df.withColumn("description", description_udf(col("name")))

        # Show the resulting DataFrame
        result_df.show(truncate=False)

        logger.info("Product descriptions generated successfully!")
    except Exception as e:
        logger.error(f"Error processing DataFrame: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    process_dataframe()
