import os
import io, base64, requests
from PIL import Image
import gradio as gr

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# Safe defaults (also read from env if you set them)
DEF_HEIGHT = int(os.getenv("SD_HEIGHT", 512))
DEF_WIDTH  = int(os.getenv("SD_WIDTH", 512))
DEF_STEPS  = int(os.getenv("SD_STEPS", 20))
DEF_GUIDE  = float(os.getenv("SD_GUIDANCE", 7.5))

def b64_to_image(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))

def generate_ui(prompt, negative, height, width, steps, guidance):
    h = int(height) if height is not None else DEF_HEIGHT
    w = int(width) if width is not None else DEF_WIDTH
    s = int(steps) if steps is not None else DEF_STEPS
    g = float(guidance) if guidance is not None else DEF_GUIDE

    payload = {
        "prompt": prompt,
        "negative_prompt": negative or None,
        "height": h,
        "width": w,
        "num_inference_steps": s,
        "guidance_scale": g,
    }
    r = requests.post(f"{API_BASE}/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    img = b64_to_image(data["image_base64"])
    score = float(data.get("clip_score", 0.0))
    score_txt = f"**CLIP score:** {score:.3f}"
    return img, score_txt

# Use a built-in theme (no CSS) and bump overall text size
theme = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")

with gr.Blocks(title="Imagen", theme=theme) as demo:
    # Title + tagline (Markdown heading is bigger by default)
    gr.Markdown("# Imagen ✨")
    gr.Markdown("Create high-quality AI images from simple text prompts.")

    # Center the whole interface by sandwiching content with spacer columns
    with gr.Row():
        gr.Column(scale=1)  # left spacer
        with gr.Column(scale=3):  # main content
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(label="Prompt", value="a watercolor painting of a fox in a misty forest")
                    negative = gr.Textbox(label="Negative prompt", placeholder="blurry, low quality")
                    height = gr.Slider(256, 1024, value=512, step=64, label="Height")
                    width  = gr.Slider(256, 1024, value=512, step=64, label="Width")
                    steps  = gr.Slider(5, 75, value=20, step=1, label="Steps")
                    guidance = gr.Slider(0, 20, value=7.5, step=0.5, label="Guidance scale")
                    btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    out_img = gr.Image(label="Result", type="pil")
                    clip_md = gr.Markdown("") 

            btn.click(generate_ui, [prompt, negative, height, width, steps, guidance], [out_img, clip_md])
        gr.Column(scale=1)  # right spacer

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("UI_SERVER_PORT", 7860)), share=True)
