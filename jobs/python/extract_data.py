from pyspark.sql import SparkSession
 
# Initialize Spark session
spark_session = SparkSession.builder.appName("Extract Data").getOrCreate()
    
try:
        # Read CSV from local filesystem (driver node)
        df = spark_session.read.csv("/opt/airflow/data/2019-Nov.csv", header=True, inferSchema=True)
        
        # Display the first 50 rows
        df.show(50)
except Exception as e:
        print(f"Error reading CSV file: {e}")
finally:
        # Stop Spark session
        spark_session.stop()