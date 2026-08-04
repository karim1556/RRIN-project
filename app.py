"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK)

The Gradio SDK simply provides a Python runtime and expects a server on port 7860.
We skip Gradio entirely (it has a Jinja2 bug in v4.44.0) and run FastAPI/uvicorn directly.
"""

import uvicorn
from api.app import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")



