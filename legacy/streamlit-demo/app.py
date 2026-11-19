import datetime
import streamlit as st
import leafmap.foliumap as leafmap
import numpy as np
from PIL import Image
from pystac_client import Client
import planetary_computer
import stackstac
import folium
from folium.plugins import Draw
from folium.template import Template
from branca.element import MacroElement

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="GeoMata")
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
    <div class="eyebrow">GEOSPATIAL INTELLIGENCE / LAND COVER</div>
    <h1 class="hero-title">Geo<span>Mata</span></h1>
    <p class="hero-copy">Satellite imagery, sharpened into a living land-cover readout.</p>
    <div class="status-line"><span class="status-dot"></span> PLANETARY COMPUTER CONNECTED &nbsp;·&nbsp; SENTINEL-2 + ESA WORLDCOVER</div>
    """,
    unsafe_allow_html=True,
)

# --- SATELLITE STREAMING ENGINE ---
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

WORLD_COVER_COLORS = {
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (250, 0, 0),
    60: (180, 180, 180),
    70: (240, 240, 240),
    80: (0, 100, 200),
    90: (0, 150, 160),
    95: (0, 207, 117),
    100: (250, 230, 160),
}


def fetch_global_land_cover(bbox, target_shape=None):
    """Fetch ESA WorldCover classes and optionally align them to a target shape."""
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox)
    items = search.item_collection()
    if len(items) == 0:
        return None

    signed_items = [planetary_computer.sign(item) for item in items]
    stack = stackstac.stack(
        signed_items,
        assets=["map"],
        bounds_latlon=bbox,
        epsg=4326,
    )
    data = np.nan_to_num(stack.compute().values, nan=0).astype(np.uint8)
    land_cover = np.zeros(data.shape[-2:], dtype=np.uint8)

    for tile in data[:, 0, :, :]:
        valid = (tile > 0) & (land_cover == 0)
        land_cover[valid] = tile[valid]

    if target_shape is not None and land_cover.shape != target_shape:
        reference_image = Image.fromarray(land_cover)
        land_cover = np.asarray(
            reference_image.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )

    return land_cover


def colorize_global_land_cover(land_cover: np.ndarray) -> np.ndarray:
    """Convert ESA WorldCover class IDs into an RGB visualization."""
    colorized = np.zeros((*land_cover.shape, 3), dtype=np.uint8)
    for class_id, color in WORLD_COVER_COLORS.items():
        colorized[land_cover == class_id] = color
    return colorized


# --- AOI RECTANGLE TOOL ---
class SingleClickRectangle(MacroElement):
    """
    Changes the Leaflet.draw rectangle tool to click once for the first corner,
    move the mouse, then click again to finish. Only the latest rectangle is kept.
    Must be added to the map before the Draw control so the new tooltip text is used.
    """

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            var aoiGroup = {{ this.group.get_name() }};
            // streamlit-folium reads the drawn shapes from window.drawnItems
            window.drawnItems = aoiGroup;

            L.drawLocal.draw.toolbar.buttons.rectangle = "Draw area of interest";
            L.drawLocal.draw.handlers.rectangle.tooltip.start = "Click to place the first corner.";
            L.drawLocal.draw.handlers.simpleshape.tooltip.end = "Move the mouse, then click again to finish.";

            var rectangle = L.Draw.Rectangle.prototype;
            var addHooks = rectangle.addHooks;
            rectangle.addHooks = function () {
                this._pressActive = false;
                this._finishOnRelease = false;
                addHooks.call(this);
            };
            rectangle._onMouseDown = function (e) {
                L.DomEvent.preventDefault(e.originalEvent);
                // Leaflet fires both mousedown and touchstart for one press; only count it once
                if (this._pressActive) {
                    return;
                }
                this._pressActive = true;
                // The first click places the start corner, the second click finishes
                this._finishOnRelease = !!this._isDrawing;
                if (!this._isDrawing) {
                    this._isDrawing = true;
                    this._startLatLng = e.latlng;
                }
                L.DomEvent.on(document, "mouseup touchend", this._onMouseUp, this);
            };
            rectangle._onMouseUp = function () {
                L.DomEvent.off(document, "mouseup touchend", this._onMouseUp, this);
                this._pressActive = false;
                if (!this._finishOnRelease) {
                    return;
                }
                this._finishOnRelease = false;
                if (this._shape) {
                    this._fireCreatedEvent();
                }
                this.disable();
            };

            // Keep only the newest rectangle
            map.on(L.Draw.Event.CREATED, function (e) {
                aoiGroup.eachLayer(function (layer) {
                    if (layer !== e.layer) {
                        aoiGroup.removeLayer(layer);
                    }
                });
            });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, group):
        super().__init__()
        self._name = "SingleClickRectangle"
        self.group = group


# --- UI LAYOUT ---
# Panel 1 (map) on top, Panel 2 (pipeline + results) below, both full width
st.subheader("1. Select Area of Interest (AOI)")
# m = leafmap.Map(center=[37.7749, -122.4194], zoom=12) # Default San Francisco
m = leafmap.Map(center=[-6.2146, 106.8451], zoom=12, draw_control=False) # Default Cibinong
aoi_group = folium.FeatureGroup(name="Area of interest", control=False).add_to(m)
SingleClickRectangle(aoi_group).add_to(m)
Draw(
    feature_group=aoi_group,
    show_geometry_on_click=False,
    # Rectangle only. showArea is off because it throws an error in Leaflet.draw 1.0.2
    draw_options={
        "polyline": False,
        "polygon": False,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "rectangle": {"shapeOptions": {"color": "#54d6c5", "weight": 2, "showArea": False}},
    },
    edit_options={"edit": False, "remove": False},
).add_to(m)
map_data = m.to_streamlit(height=500, bidirectional=True)

# --- DATE CONTROLS (directly below the map) ---
year_col, date_col = st.columns([1, 3])

# Year selection dropdown (Sentinel-2 reliable coverage: 2017 to current year 2026)
selected_year = year_col.selectbox("Select Year", list(range(2017, 2027)), index=7) # Default: 2024

# Date Slider across the selected year
start_of_year = datetime.date(selected_year, 1, 1)
end_of_year = datetime.date(selected_year, 12, 31)

selected_date = date_col.slider(
    "Select Approximate Date",
    min_value=start_of_year,
    max_value=end_of_year,
    value=datetime.date(selected_year, 6, 15), # Default mid-year
    format="YYYY-MM-DD"
)

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
    
    if st.button("Fetch Area", type="primary"):
        with st.spinner(f"Searching Sentinel-2 imagery around {selected_date}..."):
            rgb_data, meta = fetch_sentinel_by_date(
                [min_lon, min_lat, max_lon, max_lat],
                target_date=selected_date
            )
            
        if rgb_data is None:
            st.session_state["results"] = {
                "error": f"No Sentinel-2 imagery (<25% clouds) found within ±20 days of {selected_date}."
            }
        else:
            with st.spinner("Loading global land-cover reference..."):
                global_land_cover = fetch_global_land_cover(
                    [min_lon, min_lat, max_lon, max_lat],
                    target_shape=rgb_data.shape[:2],
                )
            st.session_state["results"] = {
                "rgb": rgb_data,
                "meta": meta,
                "land_cover": None if global_land_cover is None else colorize_global_land_cover(global_land_cover),
            }

else:
    st.info("Use the rectangle tool on the map: click once, move, then click again to select an area.")

# Results live in session state so they survive the reruns the map triggers
results = st.session_state.get("results")
if results is not None and "error" in results:
    st.error(results["error"])
elif results is not None:
    meta = results["meta"]
    st.success(f"Acquired Scene Date: **{meta['date']}** | Clouds: **{meta['cloud_cover']:.1f}%**")

    res_col1, res_col2 = st.columns(2)
    res_col1.image(results["rgb"], caption=f"Sentinel-2 RGB ({meta['date']})", use_column_width=True)
    if results["land_cover"] is None:
        res_col2.warning("Global land-cover data is unavailable for this area.")
    else:
        res_col2.image(
            results["land_cover"],
            caption="ESA WorldCover Global Land Cover",
            use_column_width=True,
        )
