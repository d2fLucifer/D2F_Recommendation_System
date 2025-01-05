from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.providers.qdrant.operators.qdrant import QdrantIngestOperator

with DAG(
    "example_qdrant_ingest",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    # Define vectors, ids, and payload
    vectors = [
        [0.732, 0.611, 0.289],
        [0.217, 0.526, 0.416],
        [0.326, 0.483, 0.376]
    ]
    ids: list[str | int] = [32, 21, "b626f6a9-b14d-4af9-b7c3-43d8deb719a6"]
    payload = [
        {"meta": "data"},
        {"meta": "data_2"},
        {"meta": "data_3", "extra": "data"}
    ]

    # Ingest data into Qdrant
    QdrantIngestOperator(
        task_id="qdrant_ingest",
        collection_name="test_collection",
        vectors=vectors,
        ids=ids,
        payload=payload,
        batch_size=1,
        conn_id="qdrant_default",

    )


