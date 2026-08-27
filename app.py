import streamlit as st
import leafmap.foliumap as leafmap
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from pystac_client import Client
import planetary_computer
import stackstac
import math
from folium.plugins import Draw

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Local GeoAI Inference Platform")
st.title("Local GeoAI Satellite Segmentation")
st.caption("Draw a bounding box -> Stream Sentinel-2 -> Tile 1024x1024 -> SegFormer CPU Inference")

# --- MODEL LOADING (Cached so it loads only once) ---
MODEL_NAME = "ratnaonline1/segFormer-b4-city-satellite-segmentation-1024x1024"

@st.cache_resource
def load_segformer_model():
    st.write("Loading SegFormer model weights...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.eval()
    model.to("cpu")
    return processor, model

processor, model = load_segformer_model()

# --- INFERENCE ENGINE ---
def run_tile_inference(tile_rgb: np.ndarray) -> np.ndarray:
    """Passes a single (1024, 1024, 3) uint8 tile through SegFormer CPU."""
    pil_img = Image.fromarray(tile_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to("cpu")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False
    )
    predicted_mask = upsampled_logits.argmax(dim=1).squeeze(0).numpy()
    return predicted_mask.astype(np.uint8)

def process_and_tile_image(rgb_array: np.ndarray, tile_size: int = 1024) -> np.ndarray:
    """
    Tiles an arbitrary sized (H, W, 3) image into (tile_size, tile_size) patches,
    runs inference on each, and stitches the predicted masks back together.
    """
    h, w, c = rgb_array.shape
    
    # Calculate padding required to fit exact tile multiples
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size
    
    padded_img = np.pad(rgb_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    new_h, new_w, _ = padded_img.shape
    
    output_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    
    num_tiles_y = new_h // tile_size
    num_tiles_x = new_w // tile_size
    
    progress_bar = st.progress(0)
    total_tiles = num_tiles_y * num_tiles_x
    tile_count = 0
    
    st.info(f"Splitting image ({h}x{w}px) into {total_tiles} tile(s) of {tile_size}x{tile_size}...")
    
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            y_start, y_end = i * tile_size, (i + 1) * tile_size
            x_start, x_end = j * tile_size, (j + 1) * tile_size
            
            tile = padded_img[y_start:y_end, x_start:x_end, :]
            mask_tile = run_tile_inference(tile)
            output_mask[y_start:y_end, x_start:x_end] = mask_tile
            
            tile_count += 1
            progress_bar.progress(tile_count / total_tiles)
            
    # Crop mask back to original input dimensions
    return output_mask[:h, :w]

# --- SATELLITE STREAMING ENGINE ---
def fetch_sentinel_rgb(bbox):
    """Fetches Sentinel-2 RGB array from Planetary Computer for a given bbox [minx, miny, maxx, maxy]."""
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime="2024-01-01/2024-12-31",
        query={"eo:cloud_cover": {"lt": 10}}
    )
    items = search.item_collection()
    if len(items) == 0:
        return None
        
    item = planetary_computer.sign(items[0])
    stack = stackstac.stack(
        item,
        assets=["B04", "B03", "B02"],
        bounds_latlon=bbox,
        epsg=4326,
    )
    
    # Compute array in memory and format to (H, W, 3)
    data = stack.compute().values
    if data.ndim == 4:
        data = data[0]
    rgb = np.moveaxis(data, 0, -1)
    
    # Simple linear percentile stretch to convert raw reflectances to uint8 RGB (0-255)
    p2, p98 = np.percentile(rgb[~np.isnan(rgb)], (2, 98))
    rgb_scaled = np.clip((rgb - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    return rgb_scaled

# --- UI LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. Select Area of Interest (AOI)")
    m = leafmap.Map(center=[37.7749, -122.4194], zoom=12) # Default San Francisco
    Draw(export=False).add_to(m)
    map_data = m.to_streamlit(height=500, bidirectional=True)

with col2:
    st.subheader("2. Run Pipeline")
    if map_data and map_data.get("last_active_drawing"):
        bbox = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
        # Calculate bounding box [min_lon, min_lat, max_lon, max_lat]
        lons = [pt[0] for pt in bbox]
        lats = [pt[1] for pt in bbox]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        st.write(f"**Bounding Box:** `[{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]`")
        
        if st.button("Fetch & Process Area", type="primary"):
            with st.spinner("Fetching Sentinel-2 bands via STAC..."):
                rgb_data = fetch_sentinel_rgb([min_lon, min_lat, max_lon, max_lat])
                
            if rgb_data is None:
                st.error("No low-cloud Sentinel-2 imagery found for this region/date range.")
            else:
                st.success(f"Fetched Satellite Image Array Shape: {rgb_data.shape}")
                
                with st.spinner("Running SegFormer CPU Inference..."):
                    prediction_mask = process_and_tile_image(rgb_data, tile_size=1024)
                    
                st.success("Inference Complete!")
                
                # Show results side-by-side
                res_col1, res_col2 = st.columns(2)
                res_col1.image(rgb_data, caption="Sentinel-2 RGB", use_column_width=True)
                res_col2.image(prediction_mask * 40, caption="Segmentation Mask Output", use_column_width=True)
    else:
        st.info("Draw a rectangle or polygon on the map to select an area.")