from spark_session import create_spark_session

spark = create_spark_session("Extract Data")

df = spark.read.csv("s3a://huditest/2019-Nov.csv", header=True, inferSchema=True)

df.show(50)

spark.stop()

