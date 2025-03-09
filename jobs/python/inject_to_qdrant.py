# Core Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec

# Standard Python imports
import logging
from datetime import datetime

# External library imports
from qdrant_client import QdrantClient

# Custom imports
from spark_session import create_spark_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Spark session
spark = create_spark_session()

# Read CSV file
df = spark.read \
    .option("header", "true") \
    .option("mode", "PERMISSIVE") \
    .option("columnNameOfCorruptRecord", "_corrupt_record") \
    .csv("s3a://dataset/dataset1.csv")

logger.info("Successfully read CSV file")
df.show()

logger.info(f"Total records: {df.count()}")

# NLP Feature Engineering
tokenizer = Tokenizer(inputCol="name", outputCol="words")
df_tokenized = tokenizer.transform(df)

stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df_filtered = stopwords_remover.transform(df_tokenized)

word2Vec = Word2Vec(vectorSize=128, minCount=0, inputCol="filtered_words", outputCol="word2vec_features")
model = word2Vec.fit(df_filtered)
df_vectorized = model.transform(df_filtered)

to_list_udf = udf(lambda vec: vec.toArray().tolist(), ArrayType(FloatType()))
df_final = df_vectorized.withColumn("vector", to_list_udf(col("word2vec_features")))

# Select columns
df_final = df_final.select("user_id", "product_id", "name", "vector","brand","category_code","price","event_type","event_time","user_session")
df_final.show(truncate=False)

# Define COLLECTION_NAME
COLLECTION_NAME = "product_embeddings"

# Initialize Qdrant client
client = QdrantClient(url="http://103.155.161.100:6333")

# Create Qdrant collection
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={"size": 128, "distance": "Cosine"},
)

# Configure connection options for Qdrant
options = {
    "qdrant_url": "http://103.155.161.100:6334",
    "collection_name": COLLECTION_NAME,
    "embedding_field": "vector",
    "schema": df_final.schema.json(),
}

# Write DataFrame to Qdrant
df_final.printSchema()
df_final.show(truncate=False)
df_final.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .save()
print("✅ Data successfully inserted into Qdrant.")

# Stop Spark
spark.stop()
logger.info("Spark session stopped.")