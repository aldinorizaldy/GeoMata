import datetime
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
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #f2f7f7;
        --muted: #c1d1d2;
        --panel: #11191c;
        --panel-raised: #172226;
        --line: rgba(182, 210, 210, 0.15);
        --teal: #54d6c5;
        --amber: #ffb86b;
        --coral: #ff806d;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(52, 119, 112, 0.20), transparent 31rem),
            radial-gradient(circle at 95% 15%, rgba(166, 91, 59, 0.13), transparent 27rem),
            #0b1012;
        color: var(--ink);
        font-family: 'Space Grotesk', 'Avenir Next', sans-serif;
    }
    .block-container { max-width: 1480px; padding: 3rem 4rem 4rem; }
    h1, h2, h3, p, label { font-family: 'Space Grotesk', 'Avenir Next', sans-serif; }
    .stMarkdown, [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] { color: var(--ink); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    h1 { letter-spacing: 0; font-weight: 700; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    .eyebrow {
        color: var(--teal); font: 500 0.72rem 'DM Mono', monospace;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.7rem;
    }
    .hero-title { color: var(--ink); font-size: clamp(2.1rem, 4vw, 4.4rem); line-height: 0.98; margin: 0; }
    .hero-title span { color: var(--teal); }
    .hero-copy { color: var(--muted); margin: 1rem 0 2rem; max-width: 42rem; }
    .status-line { display: flex; gap: 0.65rem; align-items: center; color: var(--muted); font: 400 0.72rem 'DM Mono', monospace; }
    .status-dot { width: 0.5rem; height: 0.5rem; background: var(--teal); border-radius: 50%; box-shadow: 0 0 14px var(--teal); }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 25, 28, 0.88); border: 1px solid var(--line); border-radius: 8px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
    }
    [data-testid="stMarkdownContainer"] h3 { color: var(--ink); font-size: 1.05rem; }
    .stButton > button {
        border: 1px solid rgba(84, 214, 197, 0.45); border-radius: 5px; min-height: 3rem;
        background: var(--teal); color: #071311; font-weight: 700; transition: all 160ms ease;
    }
    .stButton > button:hover { background: #86e8db; border-color: #86e8db; transform: translateY(-1px); }
    [data-testid="stProgressBar"] > div > div { background: var(--teal); }
    [data-testid="stAlert"] { border-radius: 5px; }
    .mono { color: var(--amber); font: 400 0.78rem 'DM Mono', monospace; }
    @media (max-width: 800px) { .block-container { padding: 2rem 1rem 3rem; } .hero-title { font-size: 2.6rem; } }
    </style>
    <div class="eyebrow">GEOSPATIAL INTELLIGENCE / LOCAL INFERENCE</div>
    <h1 class="hero-title">Geo<span>Mata</span></h1>
    <p class="hero-copy">Satellite imagery, sharpened into a living land-cover readout.</p>
    <div class="status-line"><span class="status-dot"></span> PLANETARY COMPUTER CONNECTED &nbsp;·&nbsp; SEGFORMER / CPU READY</div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR DATE CONTROLS
# ==========================================
st.sidebar.header("Date Controls")

# Year selection dropdown (Sentinel-2 reliable coverage: 2017 to current year 2026)
selected_year = st.sidebar.selectbox("Select Year", list(range(2017, 2027)), index=7) # Default: 2024

# Date Slider across the selected year
start_of_year = datetime.date(selected_year, 1, 1)
end_of_year = datetime.date(selected_year, 12, 31)

selected_date = st.sidebar.slider(
    "Select Approximate Date",
    min_value=start_of_year,
    max_value=end_of_year,
    value=datetime.date(selected_year, 6, 15), # Default mid-year
    format="YYYY-MM-DD"
)

st.sidebar.info(f"Target Date: **{selected_date.strftime('%B %d, %Y')}**")

# ... [Keep your MODEL_NAME, load_segformer_model, run_tile_inference, process_and_tile_image unchanged] ...

# ==========================================
# REPLACE OLD fetch_sentinel_rgb WITH THIS:
# ==========================================
def fetch_sentinel_by_date(bbox, target_date, window_days=20):
    """
    Searches Sentinel-2 scenes within target_date ± window_days
    and returns the item closest to target_date with low cloud cover.
    """
    start_date = target_date - datetime.timedelta(days=window_days)
    end_date = target_date + datetime.timedelta(days=window_days)
    date_str = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_str,
        query={"eo:cloud_cover": {"lt": 25}} # Up to 25% clouds in window
    )
    items = search.item_collection()
    if len(items) == 0:
        return None, None
        
    # Sort closest to user's selected date
    def date_distance(item):
        item_date = item.datetime.date()
        return abs((item_date - target_date).days)
    
    sorted_items = sorted(items, key=lambda x: (date_distance(x), x.properties["eo:cloud_cover"]))
    best_item = planetary_computer.sign(sorted_items[0])
    
    actual_date_str = best_item.datetime.strftime('%Y-%m-%d')
    cloud_cover = best_item.properties['eo:cloud_cover']
    
    stack = stackstac.stack(
        best_item,
        assets=["B04", "B03", "B02"],
        bounds_latlon=bbox,
        epsg=4326,
    )
    
    data = stack.compute().values
    if data.ndim == 4:
        data = data[0]
    rgb = np.moveaxis(data, 0, -1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    
    # Percentile contrast stretch per band
    rgb_scaled = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(3):
        band = rgb[:, :, i]
        valid = band[band > 0]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, (2, 98))
            band_stretched = np.clip((band - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255)
            rgb_scaled[:, :, i] = band_stretched.astype(np.uint8)
            
    meta_info = {"date": actual_date_str, "cloud_cover": cloud_cover}
    return rgb_scaled, meta_info

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
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Converts a 2D integer class mask into a 3D colorized RGB mask."""
    # Defined color lookup table for distinct class visualization
    COLOR_PALETTE = np.array([
        [0, 0, 0],         # Class 0: Background (Black)
        [230, 25, 75],     # Class 1: Red
        [60, 180, 75],     # Class 2: Green
        [255, 225, 25],    # Class 3: Yellow
        [0, 130, 200],     # Class 4: Blue
        [245, 130, 48],    # Class 5: Orange
        [145, 30, 180],    # Class 6: Purple
        [70, 240, 240],    # Class 7: Cyan
        [240, 50, 230],    # Class 8: Magenta
        [210, 245, 60]     # Class 9: Lime
    ], dtype=np.uint8)
    
    max_index = COLOR_PALETTE.shape[0] - 1
    safe_mask = np.clip(mask, 0, max_index)
    return COLOR_PALETTE[safe_mask]

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
    # m = leafmap.Map(center=[37.7749, -122.4194], zoom=12) # Default San Francisco
    m = leafmap.Map(center=[-6.2146, 106.8451], zoom=12) # Default Cibinong
    Draw(export=False).add_to(m)
    map_data = m.to_streamlit(height=500, bidirectional=True)

with col2:
    st.subheader("2. Run Pipeline")
    if map_data and map_data.get("last_active_drawing"):
        bbox = map_data["last_active_drawing"]["geometry"]["coordinates"][0]
        lons = [pt[0] for pt in bbox]
        lats = [pt[1] for pt in bbox]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        st.write(f"**Bounding Box:** `[{min_lon:.4f}, {min_lat:.4f}, {max_lon:.4f}, {max_lat:.4f}]`")
        
        # Display selected date info
        st.write(f"**Target Date:** `{selected_date.strftime('%Y-%m-%d')}`")
        
        if st.button("Fetch & Process Area", type="primary"):
            with st.spinner(f"Searching Sentinel-2 imagery around {selected_date}..."):
                rgb_data, meta = fetch_sentinel_by_date(
                    [min_lon, min_lat, max_lon, max_lat],
                    target_date=selected_date
                )
                
            if rgb_data is None:
                st.error(f"No Sentinel-2 imagery (<25% clouds) found within ±20 days of {selected_date}.")
            else:
                # Show metadata feedback
                st.success(f"Acquired Scene Date: **{meta['date']}** | Clouds: **{meta['cloud_cover']:.1f}%**")
                
                with st.spinner("Running SegFormer CPU Inference..."):
                    prediction_mask = process_and_tile_image(rgb_data, tile_size=1024)
                    
                st.success("Inference Complete!")
                
                res_col1, res_col2 = st.columns(2)
                res_col1.image(rgb_data, caption=f"Sentinel-2 RGB ({meta['date']})", use_column_width=True)
                res_col2.image(prediction_mask * 40, caption="Segmentation Mask Output", use_column_width=True)
    else:
        st.info("Draw a rectangle or polygon on the map to select an area.")