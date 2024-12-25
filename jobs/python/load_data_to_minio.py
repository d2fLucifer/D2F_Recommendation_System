from spark_session import create_spark_session
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os

def main():
    spark_session = create_spark_session("Load Data to Minio")
    
    # Read data from the persistent S3 location
    df = spark_session.read.parquet("s3a://recommendation/processed/my_table/")
    df.show()
    
    # Repartition to a single partition
    df_single_partition = df.coalesce(1)
    
    # Define the target directory path in MinIO
    target_directory = "s3a://recommendation/transformed/user-behavior.csv"
    
    # Write the DataFrame to MinIO in CSV format as a single file
    df_single_partition.write.mode("overwrite").option("header", True).csv(target_directory)
    
    # Initialize Hadoop FileSystem
    hadoop_conf = spark_session._jsc.hadoopConfiguration()
    fs = spark_session._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    
    # List the files in the target directory
    file_status = fs.listStatus(spark_session._jvm.org.apache.hadoop.fs.Path(target_directory))
    
    # Identify the CSV file (assuming it's the only part file)
    csv_file = None
    for status in file_status:
        file_path = status.getPath().toString()
        if file_path.endswith(".csv"):
            csv_file = file_path
            break
    
    if csv_file:
        # Define the final path with a specific filename
        final_path = "s3a://recommendation/transformed/user-behavior.csv"
        
        # Move and rename the single CSV file to the final path
        fs.rename(spark_session._jvm.org.apache.hadoop.fs.Path(csv_file), spark_session._jvm.org.apache.hadoop.fs.Path(final_path))
        
        # Optionally, delete the now-empty directory
        fs.delete(spark_session._jvm.org.apache.hadoop.fs.Path(target_directory), True)
    
    spark_session.stop()

if __name__ == "__main__":
    main()
