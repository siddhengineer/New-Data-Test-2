import torch
from transformers import AutoProcessor, AutoModel
import streamlit as st # Import streamlit for caching

# Assuming constants.py defines DEVICE and SIGLIP2_MODEL_NAME
from constants import DEVICE, SIGLIP2_MODEL_NAME

# Cache the model and processor loading using Streamlit's caching mechanism
# This function will be called by both the ingestion script and the app.
# When called from the app, st.cache_resource will cache the result.
# When called from the ingestion script (run as a standard python script),
# st.cache_resource will simply be ignored.
@st.cache_resource
def get_siglip_models_and_processor(device: str):
    """Loads SigLIP-2 model and processor."""
    print(f"Loading SigLIP-2 ({SIGLIP2_MODEL_NAME}) on {device}...")
    try:
        # Load the processor and model
        processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_NAME)
        model = AutoModel.from_pretrained(SIGLIP2_MODEL_NAME).to(device)

        # For models like SigLIP, the vision and text parts are usually integrated
        # or accessible via the same model object's forward pass depending on input.
        # We'll return the main model object and processor, and the embedding
        # functions will handle the forward pass.
        vision_model = model
        text_model = model

        print("Model and processor loaded successfully.")
        return processor, vision_model, text_model

    except Exception as e:
        print(f"Error loading SigLIP-2 model or processor: {e}")
        # Provide specific error messages if possible (e.g., network issues, model not found)
        raise e # Re-raise the exception