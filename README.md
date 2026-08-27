# GeoMata

<img width="1331" height="704" alt="image" src="https://github.com/user-attachments/assets/fd5e99ab-1fd9-4208-8e33-341cbbdf5378" />

GeoMata is a local geospatial AI workspace for turning Sentinel-2 satellite imagery into land-cover predictions. Select an area on an interactive map, stream cloud-filtered imagery from Microsoft Planetary Computer, and run a SegFormer semantic-segmentation model locally on the CPU.

## What it does

- Displays an interactive map centered on San Francisco
- Lets you draw a rectangle or polygon for an area of interest
- Searches Sentinel-2 L2A imagery from 2024 with less than 10% cloud cover
- Downloads RGB bands B04, B03, and B02 through signed Planetary Computer URLs
- Tiles imagery into 1024 x 1024 patches and predicts a land-cover mask
- Shows the source image and segmentation output in a dark Streamlit interface

## Requirements

- macOS, Linux, or Windows
- Conda or Miniforge
- Python 3.10 (provided by `environment.yml`)
- Internet access for the Planetary Computer catalog, satellite assets, and the Hugging Face model

## Local installation

Run these commands from the project directory:

```bash
conda env create -f environment.yml
conda activate geomata
```

`leafmap==0.31.1` still imports `pkg_resources`, so pin Setuptools to a compatible release:

```bash
python -m pip install "setuptools<81"
```

On an Intel Mac, PyTorch may resolve to version 2.2.x because newer wheels are unavailable for macOS x86_64. That is acceptable for this project because the model is loaded with `use_safetensors=True` and runs on the CPU.

Verify the environment:

```bash
python -c "import torch, torchvision, planetary_computer, leafmap; print('GeoMata environment is ready')"
```

## Run the app

Start Streamlit from the same activated environment:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Draw an area on the map, then select **Fetch & Process Area**. The first run downloads and caches the SegFormer model, so startup and inference can take a little time on CPU.

## Troubleshooting

- `No module named pkg_resources`: run `python -m pip install "setuptools<81"`.
- `HTTP response code: 409`: install `planetary-computer` and make sure `app.py` signs the STAC item before stacking assets.
- `Cannot pick a common CRS`: keep `epsg=4326` in the `stackstac.stack` call.
- PyTorch `2.5` or newer is unavailable on Intel macOS: use the Conda environment's compatible PyTorch version rather than forcing an unsupported wheel.

## License

License information will be added later.
