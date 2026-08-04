"""
modal_app.py
============
Serverless deployment script for RetinaAI on Modal.com.
Deploys the full FastAPI backend + Web UI + PyTorch GAN model
to a permanent, auto-scaling serverless endpoint.
"""

import modal

# Define container image with all Python dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi",
        "uvicorn",
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "opencv-python-headless",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "groq",
        "pydantic",
        "python-multipart",
        "albumentations",
        "lpip"
    )
)

app = modal.App(name="retinaai-platform", image=image)

@app.function(
    cpu=2.0,
    memory=4096,
    timeout=300,
    keep_warm=1
)
@modal.asgi_app()
def web():
    from api.app import app as fastapi_app
    return fastapi_app
