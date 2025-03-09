# Import statements (unchanged)
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from qdrant_client import QdrantClient
from spark_session import create_spark_session
import logging
from pyspark.sql.functions import unix_timestamp, to_timestamp
from pyspark.sql.functions import date_format, concat_ws, lit

# Constants (unchanged)
KAFKA_BROKER = "kafka.d2f.io.vn:9092"
TOPIC = "user-behavior-events"
MONGO_URI = "mongodb://root:example@103.155.161.100:27017/recommendation_system?authSource=admin"
QDRANT_URL = "http://103.155.161.100:6333"
COLLECTION_NAME = "userbehaviors_embeddings"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark
spark = create_spark_session()

try:
    # Read MongoDB data - User Behaviors
    df_userbehaviors = spark.read.format("mongo") \
        .option("uri", MONGO_URI) \
        .option("database", "recommendation_system") \
        .option("collection", "userbehaviors") \
        .load()
    logger.info("Read data from MongoDB collection 'userbehaviors'")
    logger.info("Schema of df_userbehaviors:")
    df_userbehaviors.printSchema()
    df_userbehaviors.show()

    # df_userbehaviors = df_userbehaviors.limit(1)

    # Read MongoDB data - Products
    df_products = spark.read.format("mongo") \
        .option("uri", MONGO_URI) \
        .option("database", "recommendation_system") \
        .option("collection", "products") \
        .load()
    
    # Log the schema to debug column names
    logger.info("Schema of df_products:")
    df_products.printSchema()

   

    # Create category_code column if columns exist
    df_products = df_products.withColumn("category_code", 
                                       concat_ws(".", col("category"), col("type"))) \
                            .drop("category", "type")
    logger.info("Read data from MongoDB collection 'products' and created category_code")

    # Join DataFrames with explicit aliasing and column selection
    df_join = df_userbehaviors.alias("ub").join(
        df_products.alias("p"),
        col("ub.product_id") == col("p.product_id"),
        "inner"
    ).select(
        col("ub.user_id"),
        col("ub.product_id").alias("product_id"), 
        col("ub.event_type"),
        col("ub.event_time"),
        col("ub.user_session"),
        col("p.name").alias("product_name"),
        col("p.brand"),
        col("p.category_code"),
        col("p.price")
    )

    logger.info("Schema of df_join:")
    df_join.printSchema()

    # NLP Feature Engineering
    tokenizer = Tokenizer(inputCol="product_name", outputCol="words")
    df_tokenized = tokenizer.transform(df_join)
    logger.info("Tokenization completed")

    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_filtered = stopwords_remover.transform(df_tokenized)
    logger.info("Stopwords removal completed")

    word2vec = Word2Vec(vectorSize=128, minCount=0, inputCol="filtered_words", outputCol="word2vec_features")
    model = word2vec.fit(df_filtered)
    df_vectorized = model.transform(df_filtered)
    logger.info("Word2Vec transformation completed")

    # Convert vector to list and select final columns
    to_list_udf = udf(lambda vec: vec.toArray().tolist(), ArrayType(FloatType()))
    df_final = df_vectorized.withColumn("vector", to_list_udf(col("word2vec_features"))) \
                            .select(
                                "user_id",
                                "ub_product_id",  # Updated to match alias
                                "product_name",   # Updated to match alias
                                "vector",
                                "brand",
                                "category_code",
                                "price",
                                "event_type",
                                "event_time",
                                "user_session"
                            )

    # Log final schema for verification
    df_final.select(
        col("user_id"),
        col("ub_product_id").alias("product_id"),
        col("product_name").alias("name"),
        col("vector"),
        col("brand"),
        col("category_code"),
        col("price"),
        col("event_type"),
        col("event_time"),
        col("user_session")

    )
    logger.info("Schema of df_final:")
    df_final = df_final.withColumn(
    "event_time",
    concat_ws(" ", date_format(col("event_time"), "yyyy-MM-dd HH:mm:ss"), lit("UTC"))
    )
    df_final.printSchema()
    df_final.show()




 
 

    # Initialize Qdrant client and create collection
    client = QdrantClient(url=QDRANT_URL)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 128, "distance": "Cosine"}
    )

    logger.info("Qdrant collection created successfully.")
    # Write to Qdrant
    options = {
        "qdrant_url": "http://103.155.161.100:6334",
        "collection_name": COLLECTION_NAME,
        "embedding_field": "vector",
        "schema": df_final.schema.json()
    }
    
    df_final.write.format("io.qdrant.spark.Qdrant") \
        .options(**options) \
        .mode("append") \
        .save()
    
    logger.info("✅ Data successfully inserted into Qdrant.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise

finally:
    # Ensure Spark session is stopped
    spark.stop()
    logger.info("Spark session stopped.")