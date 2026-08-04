"""
app.py — Entry point for Hugging Face Spaces (Gradio SDK)
Runs the FastAPI backend in a background thread on port 7861,
then serves the full Web UI inside a Gradio iframe on port 7860.
"""

import threading
import uvicorn
import gradio as gr
from api.app import app as fastapi_app

BACKEND_PORT = 7861

def _run_backend():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=BACKEND_PORT, log_level="info")

# Start the FastAPI backend in a background daemon thread
t = threading.Thread(target=_run_backend, daemon=True)
t.start()

# Full-page iframe serving the RetinaAI Web UI via the backend
html_content = f"""
<iframe 
  src="http://localhost:{BACKEND_PORT}/" 
  style="width:100%; height:95vh; border:none; margin:0; padding:0;"
  allow="camera; microphone">
</iframe>
"""

with gr.Blocks(title="RetinaAI — Retinal Intelligence Platform", css="body,html,.gradio-container{margin:0;padding:0;max-width:100%!important;} footer{display:none!important;}") as demo:
    gr.HTML(html_content)

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, share=False)



