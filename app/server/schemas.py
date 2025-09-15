from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt")
    negative_prompt: Optional[str] = None
    height: Optional[int] = Field(default=None, ge=128, le=2048)
    width: Optional[int] = Field(default=None, ge=128, le=2048)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=150)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    seed: Optional[int] = Field(default=None, ge=0)

class ImageResponse(BaseModel):
    image_base64: str
    seed: int
