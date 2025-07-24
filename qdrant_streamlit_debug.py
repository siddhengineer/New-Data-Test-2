# qdrant_streamlit_debug.py
"""
This script tests how Streamlit interacts with Qdrant
by reproducing just the relevant parts of your app.
"""

import os
import sys
import traceback
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

# Test connection and collection access
def test_qdrant_connection():
    print("Testing Qdrant connection...")
    try:
        client = QdrantClient()
        print("✅ Connected to Qdrant.")
        return client
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return None

def test_collection_access(client, collection_name):
    if not client:
        return
    
    print(f"Testing access to collection '{collection_name}'...")
    try:
        # Attempt to get collection info
        info = client.get_collection(collection_name=collection_name)
        print(f"✅ Collection info retrieved. Status: {info.status}, Points: {info.points_count}")
        
        # Check vector config
        print("Testing vector configuration access...")
        
        # Method 1: Direct access via vectors attribute (new Qdrant client style)
        try:
            vecs = info.vectors
            print("✅ Method 1 (direct vectors attr): Success")
            print(f"Vectors: {list(vecs.keys())}")
        except Exception as e:
            print(f"❌ Method 1 failed: {e}")
        
        # Method 2: Access via config.params.vectors (old style)
        try:
            vecs = info.config.params.vectors
            print("✅ Method 2 (config.params.vectors): Success")
            if hasattr(vecs, "keys"):
                print(f"Vectors keys: {list(vecs.keys())}")
            else:
                print(f"Not a dict-like object: {type(vecs)}")
        except Exception as e:
            print(f"❌ Method 2 failed: {e}")
            
        # Method 3: Try to access as dictionary
        try:
            if hasattr(info, 'vectors'):
                vecs = info.vectors
                if "image_vector" in vecs:
                    print("✅ Method 3 (vectors['image_vector']): Success")
                    print(f"Image vector size: {vecs['image_vector'].size}")
                else:
                    print("❌ 'image_vector' not found in vectors")
            else:
                print("❌ No 'vectors' attribute found")
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
            
        # Print the full structure
        print("\nCollection info structure:")
        print(f"Type: {type(info)}")
        print(f"Attributes: {dir(info)}")
        print("\nFull info dump:")
        print(info)
        
        return info
    except Exception as e:
        print(f"❌ Collection access error: {e}")
        traceback.print_exc()
        return None

def test_streamlit_pattern(collection_name):
    """Test the exact pattern used in the Streamlit app."""
    client = QdrantClient()
    try:
        info = client.get_collection(collection_name=collection_name)
        print("\nReproducing the Streamlit check_qdrant_collection_ready logic...")
        
        # This is the key part that's failing in your Streamlit app
        vecs = getattr(info, 'vectors', {})
        print(f"Vectors after getattr: {type(vecs)}")
        
        if not isinstance(vecs, dict):
            print("❌ ISSUE FOUND: vectors is not a dict")
            print(f"It's a {type(vecs)} instead")
            
            # Try to examine what it is
            if hasattr(vecs, "__dict__"):
                print(f"Internal attributes: {vecs.__dict__}")
            
            # Try to see if it's dict-like
            if hasattr(vecs, "keys"):
                print(f"It has keys: {list(vecs.keys())}")
                
                # This might be your issue - try converting to dict
                vecs_dict = {k: vecs[k] for k in vecs.keys()}
                print(f"Converted to dict: {type(vecs_dict)}")
                print(f"Dict keys: {list(vecs_dict.keys())}")
                
                # Check if image_vector exists now
                if "image_vector" in vecs_dict:
                    print("✅ 'image_vector' found in the converted dict")
                else:
                    print("❌ 'image_vector' still not found after conversion")
            else:
                print("❌ Object doesn't have keys method")
        else:
            print("✅ vectors is already a dict")
            required = {"image_vector", "text_vector"}
            missing = required - set(vecs.keys())
            if missing:
                print(f"❌ Missing vectors: {missing}")
            else:
                print("✅ All required vectors present")
        
    except Exception as e:
        print(f"❌ Error in streamlit pattern test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Get collection name from command line arg or use default
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "hf_fashion_small_siglip2_naflex_test"
    
    print(f"Testing with collection: {collection_name}")
    client = test_qdrant_connection()
    if client:
        info = test_collection_access(client, collection_name)
        test_streamlit_pattern(collection_name)