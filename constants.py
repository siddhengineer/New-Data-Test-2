# constants.py
import torch

# Determine device for PyTorch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print("CUDA is available. Using GPU.")
else:
    print("CUDA not available. Using CPU.")

# Qdrant Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333 # Default Qdrant port
# CHANGED: New collection name for this specific test dataset
QDRANT_COLLECTION_NAME = "hf_fashion_small_siglip2_naflex_test"

# Hugging Face Dataset Name
# NEW: Variable to hold the Hugging Face dataset identifier
HF_DATASET_NAME = "ashraq/fashion-product-images-small"

# SigLIP-2 Model Name
# Kept the same model name
SIGLIP2_MODEL_NAME = "google/siglip2-so400m-patch16-naflex"

# Maximum text length for tokenizer
MAX_TEXT_LEN = 77  # Standard length for SigLIP-2 tokenizer