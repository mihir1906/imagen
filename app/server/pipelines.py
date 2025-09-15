import os
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional

def _pick_device() -> str:
    # Prefer GPU if present, else Apple Silicon (MPS), else CPU
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class SDPipeline:
    def __init__(self):
        model_id = os.getenv("MODEL_ID", "sd-legacy/stable-diffusion-v1-5")
        #token = os.getenv("HUGGINGFACE_TOKEN") or None

        # Allow manual override via .env (DEVICE=cpu|cuda|mps)
        device_env = (os.getenv("DEVICE") or "").strip().lower()
        self.device = device_env if device_env in {"cpu", "cuda", "mps"} else _pick_device()

        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            #token=token,  # use HF token if model requires license
            # safety_checker=None,  # uncomment to disable safety checker
        )
        self.pipe.to(self.device)

    def generate(self, *, prompt: str, negative_prompt: Optional[str], height: int,
                 width: int, steps: int, guidance: float, seed: int):
        # MPS has quirks with generator device; using CPU generator works well
        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
        return result.images[0]

# Lazy singleton
_pipe: Optional[SDPipeline] = None
def get_pipeline() -> SDPipeline:
    global _pipe
    if _pipe is None:
        _pipe = SDPipeline()
    return _pipe
