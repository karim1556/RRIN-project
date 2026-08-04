"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK)
Mounts the RetinaAI FastAPI application and Web UI cleanly on Hugging Face.
"""

import gradio as gr
from api.app import app as fastapi_app

# Mount our full FastAPI application inside Gradio
demo = gr.mount_gradio_app(
    app=fastapi_app,
    blocks=gr.Blocks(title="RetinaAI — Retinal Intelligence Platform"),
    path="/"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:demo", host="0.0.0.0", port=7860, reload=False)
