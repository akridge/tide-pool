"""Tide Pool server: serves the web app and embeds free-text / uploaded-image queries
with the same model that built the bundle, so they land in the same vector space.

    python server/app.py                  # http://localhost:8000
    python server/app.py --no-model       # static only (search falls back to class prototypes)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import embedders  # noqa: E402

WEB = ROOT / "web"
app = FastAPI(title="Tide Pool")
state: dict = {"embedder": None, "status": "not loaded", "dim": 256}


def manifest() -> dict:
    p = WEB / "data" / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_model(kind: str, model_id: str, device: str | None) -> None:
    state["status"] = "loading"
    try:
        state["embedder"] = embedders.make(kind, model_id, device)
        state["status"] = "ready"
    except Exception as e:  # surfaced through /api/health
        state["status"] = f"error: {e}"
        embedders.log(f"model load failed: {e}")


def to_web(v: np.ndarray) -> list[float]:
    v = v.reshape(-1)[: state["dim"]]  # Matryoshka truncation to the browser's dimension
    return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32).round(6).tolist()


def need_model():
    if state["embedder"] is None:
        raise HTTPException(503, f"model {state['status']}")
    return state["embedder"]


class TextQuery(BaseModel):
    text: str
    kind: str = "query"  # "query" (SearchQuery prompt) or "document"


@app.get("/api/health")
def health():
    e = state["embedder"]
    return {"status": state["status"], "model": e.name if e else None, "dim": state["dim"]}


@app.post("/api/embed_text")
def embed_text(q: TextQuery):
    e = need_model()
    return {"vector": to_web(e.texts([q.text], kind=q.kind)[0])}


@app.post("/api/embed_image")
async def embed_image(file: UploadFile = File(...)):
    e = need_model()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image")
    return {"vector": to_web(e.images([img])[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    m = manifest()
    if not m:
        sys.exit("web/data/manifest.json not found - run `python pipeline/build.py` first")
    state["dim"] = m.get("web_dim", 256)
    if not args.no_model:
        kind = "stub" if m.get("model_is_stub") else "gemma"
        model_id = embedders.MODEL_ID if kind == "stub" else m["model"]
        threading.Thread(target=load_model, args=(kind, model_id, args.device), daemon=True).start()
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    print(f"Tide Pool on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
