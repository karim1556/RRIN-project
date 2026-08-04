"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK)
Mounts the RetinaAI FastAPI application and Web UI cleanly on Hugging Face.
"""

import os
import gradio as gr
from api.app import app as fastapi_app

# Add route for iframe UI rendering inside HF Space
@fastapi_app.get("/hf-ui", include_in_schema=False)
async def serve_hf_ui():
    from fastapi.responses import HTMLResponse
    ui_path = os.path.join(os.path.dirname(__file__), "api", "index.html")
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>UI file not found</h1>", status_code=404)

# Create Gradio Blocks container exported as `demo` (required by HF Spaces)
with gr.Blocks(title="RetinaAI — Retinal Intelligence Platform", css=".gradio-container { padding: 0 !important; max-width: 100% !important; }") as demo:
    gr.HTML('<iframe src="/hf-ui" style="width:100%; height:92vh; border:none; margin:0; padding:0;"></iframe>')

# Mount Gradio app over FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

