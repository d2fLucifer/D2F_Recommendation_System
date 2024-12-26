# load_data_to_minio.py
from spark_session import create_spark_session
from pyspark.sql import SparkSession
from pyspark import SparkConf
import os

spark = create_spark_session("load_data_to_minio")

df = spark.read.csv("s3a://recommendation/processed/my_table/")

df.write.mode("append").parquet("s3a://recommendation/load_data/")

spark.stop()
