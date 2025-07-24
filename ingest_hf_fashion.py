import os
import traceback
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance
from transformers import AutoProcessor, AutoModel

from constants import (
    DEVICE,
    HF_DATASET_NAME,
    QDRANT_COLLECTION_NAME,
)

# Inline recreate function to ensure named vectors are set correctly
def recreate_qdrant_collection(image_vector_size: int, text_vector_size: int):
    client = QdrantClient()
    coll = QDRANT_COLLECTION_NAME

    # Check existing collections via get_collections()
    existing = [c.name for c in client.get_collections().collections]
    if coll in existing:
        print(f"Deleting existing collection: {coll}")
        client.delete_collection(coll)

    print(f"Creating collection: {coll} with named vectors")
    vectors_config = {
        "image_vector": VectorParams(size=image_vector_size, distance=Distance.COSINE),
        "text_vector":  VectorParams(size=text_vector_size,  distance=Distance.COSINE),
    }
    client.create_collection(
        collection_name=coll,
        vectors_config=vectors_config
    )
    print(f"Collection '{coll}' created with named vectors.")

# Main ingestion function
def ingest_hf_dataset_to_qdrant(batch_size: int = 256):
    print(f"Loading dataset: {HF_DATASET_NAME}")
    dataset = load_dataset(HF_DATASET_NAME, split="train", keep_in_memory=True)
    N = len(dataset)
    print(f"Found {N} items.")
    if N == 0:
        raise ValueError("Dataset is empty, aborting.")

    print(f"Loading SigLIP-2 processor + model on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    )
    model = AutoModel.from_pretrained(
        "google/siglip2-so400m-patch16-naflex"
    ).to(DEVICE)
    model.eval()
    print("Models loaded successfully.")

    # Determine embedding dimensions
    example = dataset[0]
    proc_example = processor(
        text=[example["productDisplayName"]],
        images=[example["image"]],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**proc_example)
    img_dim = outputs.image_embeds.shape[-1]
    txt_dim = outputs.text_embeds.shape[-1]
    print(f"Detected dims → image: {img_dim}, text: {txt_dim}")

    # Recreate the Qdrant collection
    recreate_qdrant_collection(
        image_vector_size=img_dim,
        text_vector_size=txt_dim
    )

    client = QdrantClient()
    print(f"Indexing {N} items in batches of {batch_size}…")

    # Batch-wise embedding and upsert
    for start_idx in tqdm(range(0, N, batch_size), desc="Batch Embedding & Upsert"):
        end_idx = min(start_idx + batch_size, N)
        batch = [dataset[i] for i in range(start_idx, end_idx)]
        texts = [ex["productDisplayName"] for ex in batch]
        images = [ex["image"] for ex in batch]

        proc = processor(
            text=texts,
            images=images,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(DEVICE)

        try:
            with torch.no_grad():
                out = model(**proc)

            img_embs = out.image_embeds.cpu().numpy()
            txt_embs = out.text_embeds.cpu().numpy()

            points = []
            for j, (img_emb, txt_emb) in enumerate(zip(img_embs, txt_embs)):
                idx = start_idx + j
                # Normalize embeddings
                img_emb = img_emb / (np.linalg.norm(img_emb) or 1)
                txt_emb = txt_emb / (np.linalg.norm(txt_emb) or 1)

                points.append(models.PointStruct(
                    id=idx,
                    vector={
                        "image_vector": img_emb.tolist(),
                        "text_vector": txt_emb.tolist()
                    },
                    payload={
                        "dataset_index": idx,
                        "text": texts[j]
                    }
                ))

            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points,
                wait=True
            )

        except Exception as e:
            print(f"\nError processing batch {start_idx}-{end_idx}: {e}")
            traceback.print_exc()
            continue

    print("Ingestion complete.")

if __name__ == "__main__":
    try:
        ingest_hf_dataset_to_qdrant(batch_size=256)
    except Exception as e:
        print("\nFatal error during ingestion:", e)
        traceback.print_exc()
