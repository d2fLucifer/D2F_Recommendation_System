from spark_session import create_spark_session

def main():
    spark_session = create_spark_session("Extract Data")
    
    # Read data from S3
    df = spark_session.read.csv("s3a://recommendation/raw/2019-Nov.csv", header=True, inferSchema=True)
    
    # Write data to a persistent S3 location
    df.write.mode("append").parquet("s3a://recommendation/processed/my_table/")
    
    spark_session.stop()

if __name__ == "__main__":
    main()
