"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK)
Mounts the RetinaAI FastAPI application and Web UI cleanly on Hugging Face.
"""

import os
import gradio as gr
from api.app import app as fastapi_app

# Add route for iframe UI rendering inside HF Space
@fastapi_app.get("/ui-embed", include_in_schema=False)
async def serve_embedded_ui():
    from fastapi.responses import HTMLResponse
    ui_path = os.path.join(os.path.dirname(__file__), "api", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>UI file not found</h1>", status_code=404)

# Create Gradio Blocks container exported as `demo`
with gr.Blocks(title="RetinaAI — Retinal Intelligence Platform", css=".gradio-container { padding: 0 !important; max-width: 100% !important; } footer { visibility: hidden; }") as demo:
    gr.HTML('<iframe src="/ui-embed" style="width:100%; height:95vh; border:none; margin:0; padding:0;"></iframe>')

# Mount Gradio app over FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)


