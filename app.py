"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK)
Mounts the RetinaAI FastAPI application and Web UI cleanly on Hugging Face.
"""

import os
from api.app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

