import sys
import os
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.degradation import compose_random_degradation_pipeline
from src.quality_scoring import extract_field_of_view_mask
from src.utils.image_utils import load_image_as_float_array, save_float_array_as_image

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/degrade_image.py <path_to_image>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = os.path.splitext(input_path)[0] + "_degraded.png"
    
    print(f"Loading {input_path}...")
    clean = load_image_as_float_array(input_path)
    
    # Extract FOV mask
    uint8_img = (clean * 255).astype(np.uint8)
    fov_mask = extract_field_of_view_mask(uint8_img)
    
    print("Applying degradation pipeline...")
    degraded = compose_random_degradation_pipeline(clean, fov_mask)
    
    save_float_array_as_image(degraded, output_path)
    print(f"Saved degraded image to: {output_path}")

if __name__ == "__main__":
    main()
