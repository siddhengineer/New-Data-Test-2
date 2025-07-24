import os
import streamlit as st
from PIL import Image
import numpy as np
import traceback

# Add this line immediately after importing torch to fix the known Streamlit/torch.classes issue
try:
    import torch
    torch.classes.__path__ = []
except ImportError:
    st.error("PyTorch is not installed. Please install it to run this app.")
    torch = None

# Import constants first to avoid reference errors
import constants
from constants import DEVICE, QDRANT_COLLECTION_NAME, HF_DATASET_NAME, MAX_TEXT_LEN

# Attempt to import PyTorch and Transformers model classes
try:
    from transformers import AutoProcessor, AutoModel
except ImportError as e:
    st.error(f"Required Transformers classes not installed/found. Please install them. Error: {e}")
    AutoProcessor = None
    AutoModel = None

# Add debug mode at the top
DEBUG_MODE = True

# Wrapper for the original function to add caching and debugging for model loading
@st.cache_resource
def get_cached_siglip_models_and_processor(device=DEVICE):
    """Loads SigLIP-2 model and processor."""
    if not torch or not all([AutoProcessor, AutoModel]):
        st.error("PyTorch or essential model classes not available. Cannot load models.")
        return None, None, None
        
    try:
        print(f"Loading SigLIP-2 ({constants.SIGLIP2_MODEL_NAME}) on {device}...")
        # Load the processor and model
        processor = AutoProcessor.from_pretrained(constants.SIGLIP2_MODEL_NAME)
        model = AutoModel.from_pretrained(constants.SIGLIP2_MODEL_NAME).to(device)

        # For models like SigLIP, the vision and text parts are usually integrated
        vision_model = model
        text_model = model

        print("Model and processor loaded successfully.")
        return processor, vision_model, text_model

    except Exception as e:
        st.error(f"Error loading SigLIP-2 model or processor: {e}")
        if DEBUG_MODE:
            st.error(traceback.format_exc())
        return None, None, None

# Core imports from other project files
from qdrant_ops import get_qdrant_client, search_qdrant
from datasets import load_dataset

st.set_page_config(layout="wide")

@st.cache_resource
def load_hf_dataset_cached(dataset_name: str):
    """Loads a Hugging Face dataset with caching and robust split handling."""
    try:
        dataset = load_dataset(dataset_name, split="train")
    except TypeError as e: 
        if "unexpected keyword argument 'split'" in str(e).lower():
            dataset_dict = load_dataset(dataset_name)
            if "train" in dataset_dict:
                dataset = dataset_dict["train"]
            else:
                available_splits = list(dataset_dict.keys())
                if available_splits:
                    st.warning(f"'train' split not found in {dataset_name}. Using '{available_splits[0]}' split instead.")
                    dataset = dataset_dict[available_splits[0]]
                else:
                    st.error(f"Could not load dataset '{dataset_name}': No splits found.")
                    return None
        else: 
            raise 
    except Exception as e:
        st.error(f"Could not load dataset '{dataset_name}': {e}")
        if DEBUG_MODE: st.error(traceback.format_exc())
        return None
    return dataset

def check_qdrant_collection_ready_status():
    """Checks if the Qdrant collection is ready and configured with named vectors."""
    client = get_qdrant_client()
    if client is None:
        st.error("Failed to get Qdrant client. Check Qdrant server and connection settings.")
        return False
    collection_name_str = QDRANT_COLLECTION_NAME 
    
    try:
        all_collections = client.get_collections()
        collection_names = [c.name for c in all_collections.collections]
        if collection_name_str not in collection_names:
            st.warning(f"Qdrant collection '{collection_name_str}' not found. Run the ingestion script first.")
            return False
            
        collection_info = client.get_collection(collection_name=collection_name_str)

    except Exception as e: 
        st.warning(f"Error accessing Qdrant collection '{collection_name_str}': {e}. Ensure Qdrant is running and accessible.")
        if DEBUG_MODE: st.error(traceback.format_exc())
        return False
    
    actual_vectors_config = None
    if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params') and hasattr(collection_info.config.params, 'vectors'):
        actual_vectors_config = collection_info.config.params.vectors 
    elif hasattr(collection_info, 'config') and hasattr(collection_info.config, 'vectors'): 
         actual_vectors_config = collection_info.config.vectors
    
    if actual_vectors_config is None or not isinstance(actual_vectors_config, dict):
        st.warning(f"Collection '{collection_name_str}' does not appear to use named vectors or vector config is not a dict. Please recreate. Config found: {actual_vectors_config}")
        return False

    required_vector_names = {"image_vector", "text_vector"}
    missing_vectors = required_vector_names - set(actual_vectors_config.keys())
    if missing_vectors:
        st.warning(f"Collection '{collection_name_str}' is missing required named vector configurations: {missing_vectors}. Run ingestion.")
        return False

    if collection_info.points_count == 0:
        st.warning(f"Collection '{collection_name_str}' is empty (0 points). Ingest data first.")
        return False
    
    st.success(f"Qdrant collection '{collection_name_str}' is ready with {collection_info.points_count} points and named vectors.")
    return True

@st.cache_resource
def initialize_application_resources():
    """Initializes all necessary application resources, including models and Qdrant client."""
    if torch is None:
        st.error("PyTorch is not loaded. Application cannot start.")
        return (None,) * 7 

    if not check_qdrant_collection_ready_status():
        st.error("Qdrant collection is not ready. Please check warnings and run ingestion if needed.")
        return (None,) * 7

    try:
        processor_res, vision_model_res, text_model_res = get_cached_siglip_models_and_processor(device=DEVICE)
        
        if not all([processor_res, vision_model_res, text_model_res]):
            st.error("Failed to load SigLIP2 models or processor. Application cannot proceed.")
            return (None,) * 7

        # --- Model Self-Test ---
        st.write("Performing model self-test...")
        try:
            # Create a dummy image of size 224x224 (standard size for many vision models)
            dummy_img_pil = Image.new('RGB', (224, 224), color='white')
            dummy_text_str = "a test sentence"  # Shorter text to avoid sequence length issues
            
            # Process both image and text together with strict length limits
            inputs = processor_res(
                text=[dummy_text_str],
                images=[dummy_img_pil],
                padding="max_length",
                truncation=True,
                max_length=64,  # Explicitly set to model's max_position_embeddings
                return_tensors="pt"
            ).to(DEVICE)

            # Test the model with both inputs
            with torch.no_grad():
                outputs = vision_model_res(**inputs)
                
                # Check for expected outputs
                if not (hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds')):
                    raise ValueError("Model output missing 'image_embeds' or 'text_embeds'.")
                
                # Verify embedding dimensions
                if outputs.image_embeds.shape[-1] != outputs.text_embeds.shape[-1]:
                    raise ValueError("Image and text embeddings have different dimensions.")
                
            st.success("Model test: OK")
            st.success("All models and processor initialized and self-tested successfully!")

        except Exception as model_test_e:
            st.error(f"Critical error during model/processor self-test: {str(model_test_e)}")
            if DEBUG_MODE: st.error(traceback.format_exc())
            return (None,) * 7

        qdrant_cli = get_qdrant_client()
        collection_details = qdrant_cli.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        
        vec_params_map = collection_details.config.params.vectors
        IMG_DIM_res = vec_params_map['image_vector'].size
        TXT_DIM_res = vec_params_map['text_vector'].size

        hf_dataset_res = load_hf_dataset_cached(HF_DATASET_NAME)
        if hf_dataset_res is None:
             st.warning(f"Hugging Face dataset '{HF_DATASET_NAME}' could not be loaded. Image display in results will be affected.")

        return processor_res, vision_model_res, text_model_res, qdrant_cli, IMG_DIM_res, TXT_DIM_res, hf_dataset_res
            
    except Exception as e:
        st.error(f"An error occurred during application resource initialization: {str(e)}")
        if DEBUG_MODE: st.error(traceback.format_exc())
        return (None,) * 7

# --- Main Application Logic ---
with st.spinner("Initializing application resources... This may take a moment."):
    app_resources = initialize_application_resources()
    processor, vision_model, text_model, qdrant_client, IMG_DIM, TXT_DIM, hf_dataset = app_resources

if not all([processor, vision_model, text_model, qdrant_client]):
    st.error("Core application resources (Siglip2 models, processor, or Qdrant client) failed to initialize. Please check logs. App cannot continue.")
    st.stop() 

def embed_query_image_robust(pil_image: Image.Image) -> np.ndarray:
    """Generates a normalized embedding for a query image using the Siglip2VisionModel."""
    if pil_image is None:
        st.error("Input image is None. Cannot generate embedding.")
        return None
    if vision_model is None or processor is None: # Should be caught by global check
        st.error("Vision model or processor not available for image embedding.")
        return None
        
    try:
        # First ensure image is in correct format
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Create a dummy text input since SigLIP2 requires both image and text
        dummy_text = "a photo"
        
        # Use processor to handle both image and text processing
        inputs = processor(
            text=[dummy_text],
            images=[pil_image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
            do_resize=True,
            size={"height": 224, "width": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224},
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5]
        ).to(DEVICE)
        
        if 'pixel_values' not in inputs or 'input_ids' not in inputs:
            st.error("Processor did not return required inputs")
            if DEBUG_MODE:
                st.write(f"Processor output keys: {list(inputs.keys())}")
            return None
            
        if DEBUG_MODE:
            st.write(f"Processor output keys: {list(inputs.keys())}")
            st.write(f"Pixel values shape: {inputs['pixel_values'].shape}")
            st.write(f"Input IDs shape: {inputs['input_ids'].shape}")
        
        # Process through model
        with torch.no_grad():
            output = vision_model(**inputs)
        
        embedding = None
        if hasattr(output, 'image_embeds'):
            embedding = output.image_embeds
        elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
            embedding = output.pooler_output
        elif hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
            embedding = output.last_hidden_state[:, 0]
        else:
            st.error("Vision model output does not contain expected embedding attributes.")
            if DEBUG_MODE: st.info(f"Vision model output keys: {list(output.keys()) if hasattr(output, 'keys') else 'N/A'}")
            return None
            
        embedding_np = embedding.squeeze().cpu().numpy()
        
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            st.error("Zero-vector image embedding produced.")
            return None 
        return embedding_np / norm

    except Exception as e:
        st.error(f"Error during image embedding: {str(e)}")
        if DEBUG_MODE: st.error(traceback.format_exc())
        return None

def embed_query_text_robust(text_query: str) -> np.ndarray:
    """Generates a normalized embedding for a query text using the Siglip2TextModel."""
    if not text_query or not text_query.strip():
        st.error("Input text is empty or whitespace.")
        return None
    if text_model is None or processor is None: # Should be caught by global check
        st.error("Text model or processor not available for text embedding.")
        return None

    try:
        # Create a dummy image since SigLIP2 requires both image and text
        dummy_img = Image.new('RGB', (224, 224), color='white')
        
        # Process both text and dummy image
        inputs = processor(
            text=[text_query],
            images=[dummy_img],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
            do_resize=True,
            size={"height": 224, "width": 224},
            do_center_crop=True,
            crop_size={"height": 224, "width": 224},
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5]
        ).to(DEVICE)
        
        if DEBUG_MODE:
            st.write(f"Processor output keys: {list(inputs.keys())}")
            if 'input_ids' in inputs:
                st.write(f"Input IDs shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                st.write(f"Pixel values shape: {inputs['pixel_values'].shape}")
        
        with torch.no_grad():
            output = text_model(**inputs)
        
        embedding = None
        if hasattr(output, 'text_embeds'):
            embedding = output.text_embeds
        elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
            embedding = output.pooler_output
        elif hasattr(output, 'last_hidden_state') and output.last_hidden_state is not None:
            embedding = output.last_hidden_state[:, 0]
        else:
            st.error("Text model output does not contain expected embedding attributes.")
            if DEBUG_MODE: st.info(f"Text model output keys: {list(output.keys()) if hasattr(output, 'keys') else 'N/A'}")
            return None
            
        embedding_np = embedding.squeeze().cpu().numpy()
        
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            st.error("Zero-vector text embedding produced.")
            return None
        return embedding_np / norm

    except Exception as e:
        st.error(f"Error during text embedding: {str(e)}")
        if DEBUG_MODE: st.error(traceback.format_exc())
        return None

# --- Streamlit UI ---
st.title("✨ SigLIP2 Multi-Modal Fashion Search (Import Fix)")

st.sidebar.header("Search Configuration")
search_mode = st.sidebar.radio("Search by:", ["Image", "Text", "Both (Image & Text)"], key="search_mode_radio")
results_limit_k = st.sidebar.slider("Number of results (Top K):", 1, 20, 6, key="top_k_slider")

if 'uploaded_query_image_bytes' not in st.session_state:
    st.session_state.uploaded_query_image_bytes = None
if 'query_input_text_value' not in st.session_state:
    st.session_state.query_input_text_value = ""

uploaded_query_image_display = None 

if search_mode in ("Image", "Both (Image & Text)"):
    image_file = st.sidebar.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"], key="image_uploader_widget")
    if image_file is not None:
        st.session_state.uploaded_query_image_bytes = image_file.getvalue()
        try:
            from io import BytesIO
            uploaded_query_image_display = Image.open(BytesIO(st.session_state.uploaded_query_image_bytes)).convert("RGB")
            st.sidebar.image(uploaded_query_image_display, caption="Uploaded Image", width=150)
        except Exception as e:
            st.sidebar.error(f"Failed to display uploaded image: {e}")
            st.session_state.uploaded_query_image_bytes = None 
    elif st.session_state.uploaded_query_image_bytes: 
        try:
            from io import BytesIO
            uploaded_query_image_display = Image.open(BytesIO(st.session_state.uploaded_query_image_bytes)).convert("RGB")
            st.sidebar.image(uploaded_query_image_display, caption="Current Image", width=150)
        except Exception as e:
            st.sidebar.error(f"Failed to display image from session: {e}")
            st.session_state.uploaded_query_image_bytes = None


if search_mode in ("Text", "Both (Image & Text)"):
    st.session_state.query_input_text_value = st.sidebar.text_input(
        "Enter text description:", 
        value=st.session_state.query_input_text_value, 
        placeholder="e.g., 'blue denim jacket'", 
        key="text_query_input_widget"
    )
query_input_text_to_use = st.session_state.query_input_text_value


col_reset, col_search = st.sidebar.columns(2)
if col_reset.button("Reset Query", key="reset_button_widget", use_container_width=True):
    st.session_state.uploaded_query_image_bytes = None
    st.session_state.query_input_text_value = ""
    st.experimental_rerun()


if col_search.button("🚀 Search", key="search_button_widget", type="primary", use_container_width=True):
    actual_uploaded_image_pil = None
    if st.session_state.uploaded_query_image_bytes:
        try:
            from io import BytesIO
            actual_uploaded_image_pil = Image.open(BytesIO(st.session_state.uploaded_query_image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Error processing uploaded image data: {e}")
            st.session_state.uploaded_query_image_bytes = None

    if query_input_text_to_use and len(query_input_text_to_use.split()) > MAX_TEXT_LEN : 
        st.warning(f"Input text may be truncated to {MAX_TEXT_LEN} tokens by tokenizer.")

    image_embedding_vec = None
    text_embedding_vec = None
    can_proceed_to_search = False

    if search_mode == "Image" or search_mode == "Both (Image & Text)":
        if actual_uploaded_image_pil:
            with st.spinner("Encoding image..."):
                image_embedding_vec = embed_query_image_robust(actual_uploaded_image_pil)
            if image_embedding_vec is None: st.error("Failed to generate image embedding.")
            else:
                st.success("Image encoded.")
                if search_mode == "Image": can_proceed_to_search = True
        elif search_mode == "Image": st.warning("Please upload an image for 'Image' search.")

    if search_mode == "Text" or search_mode == "Both (Image & Text)":
        if query_input_text_to_use.strip():
            with st.spinner("Encoding text..."):
                text_embedding_vec = embed_query_text_robust(query_input_text_to_use)
            if text_embedding_vec is None: st.error("Failed to generate text embedding.")
            else:
                st.success("Text encoded.")
                if search_mode == "Text": can_proceed_to_search = True
        elif search_mode == "Text": st.warning("Please enter text for 'Text' search.")
    
    if search_mode == "Both (Image & Text)":
        if image_embedding_vec is not None and text_embedding_vec is not None:
            can_proceed_to_search = True
        else:
            if not actual_uploaded_image_pil: st.warning("Upload image for 'Both' mode.")
            if not query_input_text_to_use.strip(): st.warning("Enter text for 'Both' mode.")
            can_proceed_to_search = False

    search_results_hits = []
    if can_proceed_to_search:
        st.info(f"Searching Qdrant collection '{QDRANT_COLLECTION_NAME}'...")
        try:
            if search_mode == "Image":
                search_results_hits = search_qdrant(image_embedding_vec, "image_vector", results_limit_k)
            elif search_mode == "Text":
                search_results_hits = search_qdrant(text_embedding_vec, "text_vector", results_limit_k)
            elif search_mode == "Both (Image & Text)":
                st.write("Performing multi-vector search and merging...")
                k_per = results_limit_k * 2 
                img_hits = search_qdrant(image_embedding_vec, "image_vector", k_per)
                txt_hits = search_qdrant(text_embedding_vec, "text_vector", k_per)
                
                merged_map = {}
                for h_list in [img_hits, txt_hits]:
                    if h_list:
                        for h_item in h_list:
                            pid = h_item.payload.get("dataset_index", h_item.id)
                            if pid not in merged_map:
                                merged_map[pid] = {'obj': h_item, 'scores': []}
                            merged_map[pid]['scores'].append(h_item.score)
                
                final_list = []
                for pid, data in merged_map.items():
                    data['obj'].score = max(data['scores']) if data['scores'] else 0 
                    final_list.append(data['obj'])
                search_results_hits = sorted(final_list, key=lambda x: x.score, reverse=True)[:results_limit_k]
        except Exception as search_e:
            st.error(f"Qdrant search error: {str(search_e)}")
            if DEBUG_MODE: st.error(traceback.format_exc())

        if not search_results_hits: st.info("No matching results found.")
        else:
            st.subheader(f"Top {len(search_results_hits)} Search Results:")
            num_cols = st.slider("Results columns:", 2, 5, 3, key="num_display_cols_slider")
            
            for i in range(0, len(search_results_hits), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    if i + j < len(search_results_hits):
                        hit = search_results_hits[i+j]
                        with cols[j]:
                            st.markdown(f"**Score:** {hit.score:.4f}")
                            idx = hit.payload.get("dataset_index")
                            
                            if hf_dataset and idx is not None:
                                try:
                                    idx = int(idx)
                                    if 0 <= idx < len(hf_dataset):
                                        item = hf_dataset[idx]
                                        if "image" in item and item["image"]:
                                            st.image(item["image"], use_column_width="always", caption=f"Index: {idx}")
                                        
                                        txt_payload = hit.payload.get("text", "")
                                        txt_ds = item.get("text", item.get("description", ""))
                                        disp_txt = txt_payload if txt_payload else txt_ds
                                        if disp_txt: st.caption(f"Desc: {disp_txt[:100]}{'...' if len(disp_txt) > 100 else ''}")
                                        if "productDisplayName" in item: st.markdown(f"**{item['productDisplayName']}**")
                                        if "gender" in item: st.markdown(f"_{item['gender']}_")
                                    else: st.warning(f"Index {idx} out of bounds.")
                                except (ValueError, TypeError): st.error(f"Invalid index format: {idx}")
                                except Exception as e: st.error(f"Display error for index {idx}: {e}")
                            else: 
                                st.markdown(f"**Point ID:** {hit.id}")
                                if "text" in hit.payload: st.caption(f"Desc: {hit.payload['text']}")
                            st.markdown("---")
    elif col_search.button: 
         st.warning("Please provide valid inputs for the selected search mode before searching.")

if not col_search.button: 
    st.info("Configure your search query in the sidebar and click 'Search'.")

if DEBUG_MODE:
    st.sidebar.subheader("Debug Info")
    st.sidebar.write(f"Device: {DEVICE}")
    if IMG_DIM and TXT_DIM:
        st.sidebar.write(f"Qdrant Img Dim: {IMG_DIM}, Txt Dim: {TXT_DIM}")
    if hf_dataset: st.sidebar.write(f"HF Dataset: '{HF_DATASET_NAME}', {len(hf_dataset)} items.")
    else: st.sidebar.write(f"HF Dataset '{HF_DATASET_NAME}' not loaded.")
    st.sidebar.write(f"Max Text Length (Tokenizer): {MAX_TEXT_LEN}")

