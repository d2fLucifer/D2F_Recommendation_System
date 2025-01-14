import os
from pyspark.sql import SparkSession

def create_spark_session(
        app_name="HudiApp", 
        hudi_version='0.14.0', 
        spark_version='3.4', 
        additional_configs=None,
        mongo_connector_version='10.1.1'  # Updated to a compatible version
    ):
    """
    Create and configure a SparkSession with Hudi, S3, and MongoDB support.
    
    Parameters:
    - app_name: Name of the Spark application.
    - hudi_version: Version of Hudi to use.
    - spark_version: Version of Spark.
    - additional_configs: Dictionary for additional Spark configurations.
    - mongo_connector_version: Version of MongoDB Spark Connector to use.
    
    Returns:
    - Configured SparkSession.
    """
    
    # Define the Hudi and Hadoop AWS packages
    hudi_package = f"org.apache.hudi:hudi-spark{spark_version}-bundle_2.12:{hudi_version}"
    hadoop_aws_package = "org.apache.hadoop:hadoop-aws:3.3.2"
    mongo_package = f"org.mongodb.spark:mongo-spark-connector_2.12:{mongo_connector_version}"
    
    # Combine all packages
    all_packages = f"{hudi_package},{hadoop_aws_package},{mongo_package}"
    
    # Set the required PySpark submit arguments for Hudi, Hadoop AWS, and MongoDB Spark Connector
    # Removed PYSPARK_SUBMIT_ARGS to avoid redundancy
    # os.environ["PYSPARK_SUBMIT_ARGS"] = f"--packages {all_packages} pyspark-shell"
    
    # Create Spark session with base configurations
    spark_builder = SparkSession.builder \
        .appName(app_name) \
        .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
        .config('spark.sql.extensions', 'org.apache.spark.sql.hudi.HoodieSparkSessionExtension') \
        .config('spark.sql.hive.convertMetastoreParquet', 'false') \
        .config('spark.sql.adaptive.enabled', 'true') \
        .config('spark.sql.adaptive.shuffle.targetPostShuffleInputSize', '64m') \
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'true') \
        .config('spark.executor.memory', '4g') \
        .config('spark.executor.cores', '4') \
        .config('spark.driver.memory', '4g') \
        .config("spark.sql.broadcastTimeout", "6000s")  \
        .config('spark.memory.fraction', '0.8') \
        .config('spark.memory.storageFraction', '0.2') \
        .config('spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version', '2') \
        .config("fs.s3a.endpoint", "http://minio:9000/") \
        .config("fs.s3a.access.key", "minioadmin") \
        .config("fs.s3a.secret.key", "minioadmin") \
        .config("fs.s3a.path.style.access", "true") \
        .config("fs.s3a.connection.ssl.enabled", "false") \
        .config("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.mongodb.output.uri", "mongodb://root:example@mongo:27017/recommendation_system?authSource=admin")\
        .config("spark.jars", "/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar") \
        .master("local[*]")  \
        .config("spark.jars.packages", all_packages)  # Use all_packages here
    
    # Apply additional configurations if provided
    if additional_configs:
        for key, value in additional_configs.items():
            spark_builder = spark_builder.config(key, value)
    
    spark = spark_builder.getOrCreate()

    # Additional Hadoop settings for improved performance
    spark._jsc.hadoopConfiguration().set("parquet.enable.summary-metadata", "false")
    spark._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    spark._jsc.hadoopConfiguration().set("io.compression.codecs", 
                                         "org.apache.hadoop.io.compress.SnappyCodec,"
                                         "org.apache.hadoop.io.compress.GzipCodec")
    spark._jsc.hadoopConfiguration().set("mapreduce.output.fileoutputformat.compress", "true")

    return spark
