import os
from pyspark.sql import SparkSession

# Assuming the create_spark_session function is defined as above and saved in spark_session.py
from spark_session import create_spark_session

def load_csv_to_mongo(file_path, mongo_uri, db_name, collection_name):
    try:
        # Create Spark session with MongoDB connector
        spark = create_spark_session(
            app_name="load_data_to_mongo",
            additional_configs={
                # You can add more Spark configurations here if needed
            }
        )

        # Read CSV file into Spark DataFrame
        print("Reading CSV file into Spark DataFrame...")
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        # Transform the DataFrame (e.g., convert 'price' column to double)
        print("Transforming DataFrame...")
        if 'price' in df.columns:
            df = df.withColumn("price", df["price"].cast("double"))

        # Define MongoDB write configurations
        mongo_write_config = {
            "uri": mongo_uri,
            "database": db_name,
            "collection": collection_name
        }

        # Write the DataFrame directly to MongoDB
        print("Writing DataFrame to MongoDB...")
        df.write \
            .format("mongo") \
            .mode("append") \
            .option("uri", "mongodb://root:example@mongo:27017/recommendation_system?authSource=admin") \
            .option("database", "recommendation_system") \
            .option("collection", "user_behavior") \
            .save()


        print(f"Data successfully written to MongoDB collection '{collection_name}' in database '{db_name}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    finally:
        # Stop the Spark session
        spark.stop()
        print("Spark session stopped.")

# Example usage (replace with actual values or parameters)
if __name__ == "__main__":
    file_path = "s3a://recommendation/raw/2019-Nov.csv"  # Update with the actual file path or parameter
    mongo_uri = "mongodb://root:example@mongo:27017/admin"  # Update with the actual MongoDB URI, include credentials if necessary
    db_name = "recommendation_system"
    collection_name = "user_behavior"
    load_csv_to_mongo(file_path, mongo_uri, db_name, collection_name)
