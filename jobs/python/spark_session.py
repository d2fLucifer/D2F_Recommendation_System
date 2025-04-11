import os
from pyspark.sql import SparkSession

def create_spark_session(
        app_name="HudiApp",
        hudi_version='0.14.0',
        spark_version='3.4',
        additional_configs=None,
        mongo_connector_version='10.1.1',
        kafka_version='3.4.0'
    ):
    """
    Create and configure a SparkSession with Hudi, S3, MongoDB, and Kafka support.
    """

    # Define packages
    hudi_package = f"org.apache.hudi:hudi-spark{spark_version}-bundle_2.12:{hudi_version}"
    hadoop_aws_package = "org.apache.hadoop:hadoop-aws:3.3.2"
    mongo_package = f"org.mongodb.spark:mongo-spark-connector_2.12:{mongo_connector_version}"
    kafka_package = f"org.apache.spark:spark-sql-kafka-0-10_2.12:{kafka_version}"
    all_packages = f"{hudi_package},{hadoop_aws_package},{mongo_package},{kafka_package}"

    # Create Spark session with optimized configurations
    spark_builder = SparkSession.builder \
        .appName(app_name) \
        .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
        .config('spark.sql.extensions', 'org.apache.spark.sql.hudi.HoodieSparkSessionExtension') \
        .config('spark.sql.hive.convertMetastoreParquet', 'false') \
        .config('spark.sql.adaptive.enabled', 'true') \
        .config('spark.sql.adaptive.shuffle.targetPostShuffleInputSize', '64m') \
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'true') \
        .config('spark.executor.memory', '8g') \
        .config("spark.sql.shuffle.partitions", "400") \
        .config('spark.executor.cores', '4') \
        .config('spark.driver.memory', '8g') \
        .config('spark.memory.fraction', '0.6') \
        .config('spark.memory.storageFraction', '0.4') \
        .config('spark.executor.extraJavaOptions', '-XX:+UseG1GC') \
        .config('spark.driver.extraJavaOptions', '-XX:+UseG1GC') \
        .config("spark.sql.broadcastTimeout", "6000s") \
        .config("fs.s3a.endpoint", "http://minio:9000") \
        .config("fs.s3a.access.key", os.getenv("S3_ACCESS_KEY", "minioadmin")) \
        .config("fs.s3a.secret.key", os.getenv("S3_SECRET_KEY", "minioadmin")) \
        .config("fs.s3a.path.style.access", "true") \
        .config("fs.s3a.connection.ssl.enabled", "false") \
        .config("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.mongodb.output.uri", "mongodb://root:example@mongo:27017/recommendation_system?authSource=admin") \
        .config("spark.jars", "/usr/local/airflow/spark/jars/qdrant-spark-2.3.2.jar") \
        .config("spark.kafka.bootstrap.servers", "localhost:9092") \
        .config("spark.streaming.kafka.maxRatePerPartition", "1000") \
        .config("spark.jars.packages", all_packages) \
        .master("local[*]") 

    # Apply additional configurations if provided
    if additional_configs:
        for key, value in additional_configs.items():
            spark_builder = spark_builder.config(key, value)
    
    spark = spark_builder.getOrCreate()

    # Additional Hadoop settings for performance
    spark._jsc.hadoopConfiguration().set("parquet.enable.summary-metadata", "false")
    spark._jsc.hadoopConfiguration().set("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
    spark._jsc.hadoopConfiguration().set("io.compression.codecs", 
                                         "org.apache.hadoop.io.compress.SnappyCodec,"
                                         "org.apache.hadoop.io.compress.GzipCodec")
    spark._jsc.hadoopConfiguration().set("mapreduce.output.fileoutputformat.compress", "true")

    return spark
