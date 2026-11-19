import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

MODEL_NAME = "ratnaonline1/segFormer-b4-city-satellite-segmentation-1024x1024"

# Load model and processor onto CPU
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME, use_safetensors=True)
model.eval()

def predict_city_landcover(rgb_array: np.ndarray) -> np.ndarray:
    """
    rgb_array: NumPy array of shape (Height, Width, 3) with values 0-255
    returns: NumPy array mask containing predicted class IDs
    """
    image = Image.fromarray(rgb_array.astype('uint8'))
    
    # Preprocess image for 1024x1024 SegFormer
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Resize prediction back to original tile dimensions
    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=image.size[::-1], # (Height, Width)
        mode="bilinear",
        align_corners=False
    )
    
    predicted_mask = upsampled_logits.argmax(dim=1).squeeze(0).numpy()
    return predicted_mask

# Test with a dummy 1024x1024 RGB array
dummy_satellite_rgb = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
class_mask = predict_city_landcover(dummy_satellite_rgb)
print("Output prediction shape:", class_mask.shape)