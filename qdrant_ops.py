import os
from typing import List, Dict

import numpy as np
# No tqdm here as the main upsert loop is in the ingestion script
from qdrant_client import QdrantClient, models

from constants import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME
)

# Cache the client instance
_qdrant_client: QdrantClient = None

def get_qdrant_client() -> QdrantClient:
    """Initializes and returns a Qdrant client instance."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT
            )
            # Sanity check: attempt to list collections to verify connection
            _qdrant_client.get_collections()
            print(f"Successfully connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}.")
        except Exception as e:
            print(f"Could not connect to Qdrant at {QDRANT_HOST}:{Qdrant_PORT}. Please ensure Qdrant is running.")
            raise e

    return _qdrant_client

def recreate_qdrant_collection(image_vector_size: int, text_vector_size: int):
    """
    Recreates the Qdrant collection with named vectors for image and text embeddings.
    Intended to be called by the ingestion script.
    """
    client = get_qdrant_client()
    collection_name = QDRANT_COLLECTION_NAME

    # Delete collection if it already exists
    if client.collection_exists(collection_name=collection_name):
        print(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    # Define configurations for multiple named vectors
    vectors_config: Dict[str, models.VectorParams] = {
        "image_vector": models.VectorParams(size=image_vector_size, distance=models.Distance.COSINE),
        "text_vector": models.VectorParams(size=text_vector_size, distance=models.Distance.COSINE)
    }

    # Create the new collection with named vectors
    print(f"Creating collection: {collection_name} with vectors_config: {vectors_config}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config
    )
    print(f"Collection '{collection_name}' created successfully.")


def search_qdrant(query_embedding: np.ndarray, query_vector_name: str, top_k: int = 5):
    """Performs a search against a specific named vector in the Qdrant collection."""
    client = get_qdrant_client()
    collection_name = QDRANT_COLLECTION_NAME
    print(f"Searching collection '{collection_name}' with query_vector_name='{query_vector_name}'...")
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=models.NamedVector( # Use NamedVector for searching
                name=query_vector_name,
                vector=query_embedding.tolist() # Ensure query_embedding is a list of floats
            ),
            limit=top_k,
            with_payload=True # Retrieve payload along with search results
        )
        print(f"Found {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error during Qdrant search on collection '{collection_name}': {e}")
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            if isinstance(collection_info.vectors_config, models.VectorsConfigMap):
                 if query_vector_name not in collection_info.vectors_config.params_map:
                      print(f"Error: Query vector name '{query_vector_name}' does not exist in collection '{collection_name}'.")
                      print(f"Available vector names: {list(collection_info.vectors_config.params_map.keys())}")
            else:
                 print(f"Error: Collection '{collection_name}' is not configured with named vectors as expected.")

        except Exception as info_e:
             print(f"Could not retrieve collection info for detailed error checking: {info_e}")

        raise e