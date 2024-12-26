import asyncio
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, mean, stddev, collect_list, struct
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding

def to_sparse_vector(features, size):
    vector = np.zeros(size)
    for f in features:
        vector[int(f.product_id)] = f.normalized_interaction
    return vector.tolist()

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("transform_to_qdrant").getOrCreate()

    # Read data from the persistent S3 location
    df = spark.read.parquet("s3a://recommendation/transformed/user-behavior.csv")

    # Data preprocessing
    df = df.dropna()
    df = df.filter(df.event_type.isin('purchase', 'view'))

    interaction_df = df.withColumn("interaction", lit(1)) \
        .groupBy("user_id", "product_id") \
        .sum("interaction") \
        .withColumnRenamed("sum(interaction)", "interaction_count")

    mean_interaction = interaction_df.select(mean(col("interaction_count"))).first()[0]
    stddev_interaction = interaction_df.select(stddev(col("interaction_count"))).first()[0]

    interaction_df = interaction_df.withColumn(
        "normalized_interaction",
        (col("interaction_count") - mean_interaction) / stddev_interaction
    )

    product_count = df.select("product_id").distinct().count()

    user_vectors = interaction_df.groupBy("user_id").agg(
        collect_list(struct("product_id", "normalized_interaction")).alias("features")
    )

    user_vectors = user_vectors.rdd.map(lambda row: (
        row.user_id,
        to_sparse_vector(row.features, product_count)
    )).toDF(["user_id", "vector"])

    # Initialize Qdrant client
    client = QdrantClient(host="qdrant", port=6333)

    # Set the embedding model
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Recreate collection with the specified vector parameters
    client.recreate_collection(
        collection_name='user_preferences',
        vectors_config=VectorParams(size=product_count, distance=Distance.COSINE)
    )

    # Batch upsert points to Qdrant
    batch_size = 1000
    points = [
        PointStruct(
            id=int(row.user_id),
            vector=row.vector,
            payload={'user_id': row.user_id}
        )
        for row in user_vectors.collect()
    ]

    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name='user_preferences',
            points=points[i:i + batch_size]
        )

    # Example user preferences
    user_preferences = {'product_id_1': 1, 'product_id_2': -1}  # Replace with actual data
    user_vector = np.zeros(product_count)
    for product_id, rating in user_preferences.items():
        product_idx = int(product_id)  # Ensure product_id is an integer
        user_vector[product_idx] = rating

    # Generate embedding for the user vector
    user_vector_embedding = embedding_model.embed([user_vector.tolist()])[0]

    # Search for similar users
    search_result = client.search(
        collection_name='user_preferences',
        query_vector=user_vector_embedding,
        limit=10
    )

    # Collect recommended items
    recommended_items = set()
    for result in search_result:
        similar_user_id = result.payload['user_id']
        similar_user_items = df.filter(df.user_id == similar_user_id).select("product_id").rdd.flatMap(lambda x: x).collect()
        recommended_items.update(similar_user_items)

    # Exclude items the user has already interacted with
    user_id = 123  # Replace with the actual user_id
    user_items = df.filter(df.user_id == user_id).select("product_id").rdd.flatMap(lambda x: x).collect()
    recommended_items = recommended_items - set(user_items)

    print(f"Recommended items for user {user_id}: {recommended_items}")

    # Close the Qdrant client connection
    client.close()

if __name__ == "__main__":
    main()
