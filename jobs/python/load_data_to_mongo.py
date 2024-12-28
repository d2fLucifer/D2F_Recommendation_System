import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnull

# Assuming the create_spark_session function is defined as above and saved in spark_session.py
from spark_session import create_spark_session

def load_csv_to_mongo(file_path, mongo_uri, db_name, collection_name):
    """
    Load data from a CSV file into MongoDB using Spark.
    """
    try:
        # Create Spark session with MongoDB connector
        spark = create_spark_session(app_name="LoadCSVToMongo")

        # Read CSV file into Spark DataFrame
        print("Reading CSV file into Spark DataFrame...")
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        # Log the initial row count
        initial_count = df.count()
        print(f"Initial row count: {initial_count}")

        # Check schema
        print("Schema of the DataFrame:")
        df.printSchema()

        # Validate and clean 'price' column
        if 'price' in df.columns:
            print("Validating 'price' column...")
            invalid_price_df = df.filter(~col("price").rlike("^\d+(\.\d+)?$"))
            invalid_count = invalid_price_df.count()
            if invalid_count > 0:
                print(f"Number of rows with invalid 'price' values: {invalid_count}")
                # Optionally, handle invalid prices
                df = df.filter(col("price").rlike("^\d+(\.\d+)?$"))

            print("Transforming 'price' column to double...")
            df = df.withColumn("price", col("price").cast("double"))

        # Check for nulls
        print("Checking for null values in DataFrame...")
        null_counts = df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).collect()[0].asDict()
        print("Null values per column:", null_counts)

        # Remove rows with null values in critical columns
        critical_columns = ['price']  # Add other critical columns as needed
        print(f"Filtering rows with nulls in columns: {critical_columns}...")
        df = df.dropna(subset=critical_columns)
        filtered_count = df.count()
        print(f"Row count after filtering nulls: {filtered_count}")

        # Define MongoDB write configurations
        print("Defining MongoDB write configurations...")
        mongo_write_config = {
            "uri": mongo_uri,
            "database": db_name,
            "collection": collection_name
        }

        # Write DataFrame to MongoDB
        print("Writing DataFrame to MongoDB...")
        df.write \
            .format("mongo") \
            .mode("overwrite") \
            .options(**mongo_write_config) \
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
    file_path = "s3a://recommendation/raw/dataset.csv"  # Update with the actual file path or parameter
    mongo_uri = "mongodb://root:example@mongo:27017/admin"  # Update with the actual MongoDB URI, include credentials if necessary
    db_name = "recommendation_system"
    collection_name = "user_behavior"
    load_csv_to_mongo(file_path, mongo_uri, db_name, collection_name)
