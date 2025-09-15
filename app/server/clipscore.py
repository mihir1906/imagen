import os
from typing import Optional
import torch
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# Try new processor first; fall back if your transformers version is older
try:
    from transformers import CLIPImageProcessor as _ImgProc
except Exception:  # fallback for older versions
    from transformers import CLIPProcessor as _ImgProc  # type: ignore

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class CLIPScorer:
    def __init__(self):
        device_env = (os.getenv("DEVICE") or "").strip().lower()
        self.device = device_env if device_env in {"cpu", "cuda", "mps"} else _pick_device()

        clip_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(clip_id).to(self.device)
        self.model.eval()
        self.tok = CLIPTokenizer.from_pretrained(clip_id)
        self.proc = _ImgProc.from_pretrained(clip_id)

    def score(self, image: Image.Image, prompt: str) -> float:
        # Ensure RGB and model-friendly tensors
        image = image.convert("RGB")
        txt = self.tok([prompt], return_tensors="pt", truncation=True)
        img = self.proc(images=image, return_tensors="pt")

        # Move to device (float32)
        txt = {k: v.to(self.device) for k, v in txt.items()}
        img = {k: v.to(self.device) for k, v in img.items()}

        with torch.no_grad():
            t = self.model.get_text_features(**txt)
            v = self.model.get_image_features(**img)

        t = F.normalize(t, dim=-1)
        v = F.normalize(v, dim=-1)
        cos = (t * v).sum(dim=-1).item()           # in [-1, 1]
        return float((cos + 1.0) / 2.0)            # map to [0, 1]

_scorer: Optional[CLIPScorer] = None
def get_clip_scorer() -> CLIPScorer:
    global _scorer
    if _scorer is None:
        _scorer = CLIPScorer()
    return _scorer
