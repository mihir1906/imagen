import base64
from io import BytesIO
from PIL import Image
import random

from typing import Optional

def to_base64(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def resolve_seed(seed: Optional[int]) -> int:
    return seed if seed is not None else random.randint(0, 2**31 - 1)
