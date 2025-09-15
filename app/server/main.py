import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import GenerateRequest, ImageResponse
from .pipelines import get_pipeline
from .utils import to_base64, resolve_seed
from .clipscore import get_clip_scorer

app = FastAPI(title="Imagen API", version="0.1.0")

# CORS
origins = os.getenv("API_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate_stub(req: GenerateRequest):
    try:
        height = req.height or int(os.getenv("SD_HEIGHT", 512))
        width = req.width or int(os.getenv("SD_WIDTH", 512))
        steps = req.num_inference_steps or int(os.getenv("SD_STEPS", 30))
        guidance = req.guidance_scale or float(os.getenv("SD_GUIDANCE", 6.5))
        seed = resolve_seed(req.seed)

        pipe = get_pipeline()
        img = pipe.generate(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=height,
            width=width,
            steps=steps,
            guidance=guidance,
            seed=seed,
        )

         # Compute CLIP score (but never fail the request if this errors)
        clip_score = 0.0
        try:
            scorer = get_clip_scorer()
            clip_score = scorer.score(img, req.prompt)
        except Exception as e:
            print(f"[WARN] CLIP scoring failed: {e}")

        return ImageResponse(image_base64=to_base64(img), seed=seed, clip_score=clip_score)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
