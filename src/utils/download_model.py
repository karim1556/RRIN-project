"""
src/utils/download_model.py
===========================
Helper utility to automatically download model weights on server startup
if checkpoints/best.pt or checkpoints/best_inference.pt are missing.
"""

import os
import urllib.request
from pathlib import Path

# Default release URL (can be overridden via environment variable MODEL_URL)
DEFAULT_MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/karim1556/RRIN-project/releases/download/v1.0.0/best_inference.pt"
)

def ensure_checkpoint_exists(target_path: str = "checkpoints/best.pt") -> str:
    """
    Check if a checkpoint exists locally. If not, check for best_inference.pt.
    If neither exists, attempt to download best_inference.pt from DEFAULT_MODEL_URL or MODEL_URL.

    returns: path to existing or downloaded checkpoint file
    """
    if os.path.exists(target_path):
        return target_path

    checkpoint_dir = os.path.dirname(target_path) or "checkpoints"
    alt_path = os.path.join(checkpoint_dir, "best_inference.pt")
    if os.path.exists(alt_path):
        return alt_path

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    download_url = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)

    print(f"Downloading model weights from {download_url} to {alt_path}...")
    try:
        urllib.request.urlretrieve(download_url, alt_path)
        print(f"Successfully downloaded model weights to {alt_path}")
        return alt_path
    except Exception as e:
        print(f"Failed to download model weights from {download_url}: {e}")
        return target_path
