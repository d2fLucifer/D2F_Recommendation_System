from spark_session import create_spark_session
import logging
from pyspark.sql.functions import regexp_replace


def clean_column_names(df):
    for col_name in df.columns:
        new_col_name = (
            col_name.replace(".", "_")  # Replace dot notation
            .replace("[", "_")  # Replace opening bracket
            .replace("]", "")  # Remove closing bracket
        )
        df = df.withColumnRenamed(col_name, new_col_name)
    return df


spark = create_spark_session("Load dataset to MongoDB")

dataset_paths = {
    "carts": "s3a://dataset/recommendation_system.carts.csv",
    "products": "s3a://dataset/recommendation_system.products.csv",
    "users": "s3a://dataset/recommendation_system.users.csv",
    "types": "s3a://dataset/recommendation_system.types.csv",
    "categories": "s3a://dataset/recommendation_system.categories.csv",
    "orders": "s3a://dataset/recommendation_system.orders.csv",
    "reviews": "s3a://dataset/recommendation_system.reviews.csv",
    "shippingaddresses": "s3a://dataset/recommendation_system.shippingaddresses.csv",
    "userbehaviors": "s3a://dataset/recommendation_system.userbehaviors.csv",
}

for collection, path in dataset_paths.items():
    df = spark.read.csv(path, header=True, inferSchema=True)
    df = clean_column_names(df)  # Apply transformation
    df.write.format("mongo").mode("overwrite").option("database", "recommendation_system").option("collection", collection).save()
    logging.info(f"{collection} dataset has been loaded to MongoDB")
