# qdrant_diagnostic.py - Run this script to check your Qdrant collection status

from qdrant_client import QdrantClient
import json

# Create a client - using default connection parameters
client = QdrantClient()

def inspect_collections():
    print("=== Listing All Collections ===")
    collections = client.get_collections()
    
    if not collections.collections:
        print("No collections found!")
        return
    
    print(f"Found {len(collections.collections)} collections:")
    for coll in collections.collections:
        print(f"- {coll.name}")
    
    # Ask user which collection to inspect in detail
    coll_name = input("\nEnter collection name to inspect in detail (or press Enter to inspect the first one): ")
    if not coll_name and collections.collections:
        coll_name = collections.collections[0].name
    
    if coll_name:
        inspect_collection(coll_name)

def inspect_collection(collection_name):
    print(f"\n=== Inspecting Collection: {collection_name} ===")
    try:
        # Get collection info
        info = client.get_collection(collection_name=collection_name)
        points_count = info.points_count
        
        print(f"Points count: {points_count}")
        print(f"Status: {info.status}")
        
        # Check vector configuration
        print("\nVector configuration:")
        vectors_config = info.config.params.vectors
        if hasattr(vectors_config, "keys"):  # Named vectors
            print("Collection uses NAMED vectors:")
            for name, config in vectors_config.items():
                print(f"  - {name}: size={config.size}, distance={config.distance}")
        else:  # Single vector
            print("Collection uses a SINGLE vector:")
            print(f"  - size={vectors_config.size}, distance={vectors_config.distance}")
        
        # Get sample points if any exist
        if points_count > 0:
            print("\nSample points:")
            points = client.scroll(
                collection_name=collection_name,
                limit=3,
                with_vectors=True
            )[0]
            
            for i, point in enumerate(points):
                print(f"\nPoint {i+1} (ID: {point.id}):")
                
                # Print payload
                print("  Payload:")
                for k, v in point.payload.items():
                    print(f"    {k}: {v}")
                
                # Check vector structure
                print("  Vector structure:")
                if isinstance(point.vector, dict):  # Named vectors
                    for vec_name, vec_value in point.vector.items():
                        vec_len = len(vec_value) if vec_value else 0
                        print(f"    {vec_name}: {vec_len} dimensions")
                else:  # Single vector
                    vec_len = len(point.vector) if point.vector else 0
                    print(f"    (unnamed): {vec_len} dimensions")
        
    except Exception as e:
        print(f"Error inspecting collection: {e}")

if __name__ == "__main__":
    print("Qdrant Diagnostic Tool")
    print("=====================")
    inspect_collections()