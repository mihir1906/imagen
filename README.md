# Imagen — Client–Server Architecture

This project is designed as a **client–server system**:
- A **Server** (FastAPI) hosts the Stable Diffusion pipeline and the CLIP scorer.
- One or more **Clients** send prompts to the server and display results (Gradio UI, CLI, or programmatic calls).

---

## Features
- **Server API**: `POST /generate` → returns `{ image_base64, seed, clip_score }`
- **Client UI**: Gradio front-end with prompt, size, steps, guidance
- **CLIP score**: prompt–image similarity shown under the output image
- **GPU compatibility**: runs on **CPU** everywhere; supports **Apple Silicon (MPS)** natively on macOS and **NVIDIA CUDA GPUs** on compatible systems

---

## Components

- **Server (FastAPI)**
  - Endpoint: `POST /generate`
  - Health: `GET /health`
  - Models: Hugging Face **Diffusers** (Stable Diffusion), **PyTorch** (CPU/MPS/CUDA)
  - Scoring: **CLIP** (prompt–image similarity)
- **Clients**
  - **Gradio UI** (`app/ui/gradio_app.py`) — human-friendly front-end
  - **CLI Client** (`clients/python_client.py`) — scriptable usage (optional)
  - **Programmatic** calls via `curl` or Python `requests`

  ## Requirements
- Python **3.9+** 
- macOS / Windows / Linux
- Internet access (first run downloads model weights)

---

## Tech stack
- **FastAPI** (API) + **Uvicorn** (ASGI server)
- **Hugging Face Diffusers** (Stable Diffusion)
- **Transformers / CLIP** (scoring)
- **PyTorch** (CPU / MPS / CUDA)
- **Gradio** (UI)
- **Pydantic** (request/response models)
- **Pillow** (image handling)

---

##  Project layout (server/client focus)

~~~
imagen/
├─ app/
│  ├─ server/
│  │  ├─ __init__.py
│  │  ├─ main.py          # FastAPI routes (/health, /generate)
│  │  ├─ schemas.py       # Pydantic models for request/response
│  │  ├─ pipelines.py     # Diffusers pipeline (model/device)
│  │  ├─ utils.py         # base64 encode, seed helper
│  │  └─ clipscore.py     # CLIP model + scoring
│  └─ ui/
│     └─ gradio_app.py    # Client UI calling the server
├─ clients/
│  └─ python_client.py    # Optional CLI client
├─ requirements-api.txt
├─ requirements-ui.txt
├─ .env.example
├─ .gitignore
└─ README.md
~~~

---

## Setup

Clone your repo and create a virtual env:

~~~bash
python3 -m venv .venv
source .venv/bin/activate

# API deps
pip install -r requirements-api.txt
# UI deps
pip install -r requirements-ui.txt
~~~

_(If the model requires auth: `huggingface-cli login` or set `HUGGINGFACE_TOKEN` in `.env`.)_

---

## Configuration (.env)

Copy the example and edit if needed:

~~~bash
cp .env.example .env
~~~

Key variables:

~~~
MODEL_ID=runwayml/stable-diffusion-v1-5
DEVICE=                 # leave blank to auto; options: cpu | mps | cuda
SD_HEIGHT=512
SD_WIDTH=512
SD_STEPS=25
SD_GUIDANCE=7.5
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=*
API_BASE_URL=http://localhost:8000   # used by UI
~~~

> On macOS, set `DEVICE=mps` to use Apple GPU; otherwise it auto-detects.

---

##  Run

### Start the **Server** (FastAPI)
~~~bash
uvicorn app.server.main:app --host 0.0.0.0 --port 8000
~~~

### Start the **Client UI** (Gradio)
~~~bash
# new terminal
export API_BASE_URL=http://localhost:8000
python app/ui/gradio_app.py
~~~

Open: http://localhost:7860

---

##  Gradio UI

Type a **prompt**, optionally add a **negative prompt**, adjust **Height**, **Width**, **Steps**, and **Guidance**, then click **Generate**.  
The output image appears on the right with the **CLIP score** displayed beneath it.

*UI screenshot to be added here.*


##  Results

**Experiment:** *Batched best-of-K (micro-batched)*

**Setup summary**

| Setting            | Value                     |
|--------------------|---------------------------|
| Prompts            | 24                        |
| K per prompt       | 5                         |
| Batch size         | 2                         |
| GPU (device)       | NVIDIA **Tesla T4** (~15 GB) |
| Evaluation metric  | CLIPScore (best-of-K)     |

**Throughput & quality**

| Metric                          | Value                |
|---------------------------------|----------------------|
| Latency per image (sec)         | **5.235**+<br>−**0.053**  |
| Images per minute               | **11.46**            |
| Best-of-K CLIPScore per prompt  | **67.45**+<br>−**1.18**   |
| GPU peak per micro-batch (GB)   | **12.23**            |

> Notes: Measurements were recorded over **24** prompts on an **NVIDIA Tesla T4** (~15 GB) system.


## License (MIT)

This project is licensed under the **MIT License**. See the [`LICENSE`](./LICENSE) file for the full text.


##  Acknowledgments

Inspired by prior community work on:

- **Stable Diffusion** — [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)  
- **Hugging Face Diffusers** — <https://github.com/huggingface/diffusers>  
- **Gradio** — <https://www.gradio.app/>  

Thanks to the open-source ecosystem for making rapid prototyping possible.
