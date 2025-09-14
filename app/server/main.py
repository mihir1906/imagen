import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Imagen API", version="0.0.1")

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
def generate_stub():
    # We'll wire this up to Stable Diffusion next step.
    raise HTTPException(status_code=501, detail="Not implemented yet. Coming next step!")
