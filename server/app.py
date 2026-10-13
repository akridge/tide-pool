"""Tide Pool server: serves the web app and embeds free-text / uploaded-image queries
with the same model that built the bundle, so they land in the same vector space.

    python server/app.py                  # http://localhost:8000
    python server/app.py --no-model       # static only (search falls back to class prototypes)

Serves the sample explorer at / (from pipeline/build.py) and, when pipeline/full.py has run,
the all-patches viewer at /full/, whose similarity search runs here over every vector.
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
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import embedders  # noqa: E402

WEB = ROOT / "web"
FULL_META = WEB / "full" / "data" / "meta.json"
FULL_SEARCH = ROOT / "data" / "full" / "search.npy"
app = FastAPI(title="Tide Pool")
state: dict = {"embedder": None, "status": "not loaded", "dim": 256, "full": None}


def manifest() -> dict:
    p = WEB / "data" / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def full_meta() -> dict:
    return json.loads(FULL_META.read_text()) if FULL_META.exists() else {}


def load_full() -> None:
    """All-patches search matrix (n x web_dim, unit rows; failed downloads are zero rows)."""
    if FULL_SEARCH.exists() and FULL_META.exists():
        state["full"] = np.load(FULL_SEARCH).astype(np.float32)
        state["full_model"] = full_meta().get("model")
        embedders.log(f"full search: {len(state['full']):,} vectors")


def load_model(kind: str, model_id: str, device: str | None) -> None:
    state["status"] = "loading"
    try:
        state["embedder"] = embedders.make(kind, model_id, device)
        state["status"] = "ready"
    except Exception as e:  # surfaced through /api/health
        state["status"] = f"error: {e}"
        embedders.log(f"model load failed: {e}")


def to_web(v: np.ndarray, dim: int | None = None) -> np.ndarray:
    v = v.reshape(-1)[: dim or state["dim"]]  # Matryoshka truncation to the target dimension
    return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)


def full_model():
    """The loaded model, if it is the one that embedded the full set (else queries would be meaningless)."""
    e = need_model()
    if state["full"] is not None and e.name != state.get("full_model"):
        raise HTTPException(409, f"/full/ was embedded with {state.get('full_model')} but the server loaded "
                                 f"{e.name}; rebuild one of them with the same model")
    return e


def full_scores(q: np.ndarray) -> Response:
    """Cosine similarity of q to every patch, as little-endian float16 bytes (2 bytes x n)."""
    F = state["full"]
    if F is None:
        raise HTTPException(503, "full-dataset search matrix not loaded (run pipeline/full.py)")
    return Response((F @ to_web(q, F.shape[1])).astype("<f2").tobytes(), media_type="application/octet-stream")


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
    F = state["full"]
    return {"status": state["status"], "model": e.name if e else None, "dim": state["dim"],
            "full": None if F is None else len(F)}


@app.post("/api/embed_text")
def embed_text(q: TextQuery):
    e = need_model()
    return {"vector": to_web(e.texts([q.text], kind=q.kind)[0]).round(6).tolist()}


@app.post("/api/embed_image")
async def embed_image(file: UploadFile = File(...)):
    e = need_model()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image")
    return {"vector": to_web(e.images([img])[0]).round(6).tolist()}


class FullQuery(BaseModel):
    text: str | None = None
    index: int | None = None


@app.post("/api/full/search")
def full_search(q: FullQuery):
    """Scores for every patch against a text query or against patch `index` (find similar)."""
    if q.index is not None:
        F = state["full"]
        if F is None or not 0 <= q.index < len(F):
            raise HTTPException(400 if F is not None else 503, "bad index or matrix not loaded")
        return full_scores(F[q.index])
    if not q.text:
        raise HTTPException(400, "give text or index")
    return full_scores(full_model().texts([q.text], kind="query")[0])


@app.post("/api/full/search_image")
async def full_search_image(file: UploadFile = File(...)):
    e = full_model()
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "not an image")
    return full_scores(e.images([img])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    m, fm = manifest(), full_meta()
    if not m and not fm:
        sys.exit("no data yet - run `python pipeline/build.py` (sample) or `python pipeline/full.py` (all patches)")
    if m and fm and m.get("model") != fm.get("model"):
        embedders.log(f"note: / was built with {m.get('model')} and /full/ with {fm.get('model')}; "
                      "search uses the model of /")
    src = m or fm
    state["dim"] = src.get("web_dim", 256)
    if fm:
        threading.Thread(target=load_full, daemon=True).start()
    if not args.no_model:
        kind = "stub" if src.get("model_is_stub") else "gemma"
        model_id = embedders.MODEL_ID if kind == "stub" else src.get("model_id") or src["model"]
        threading.Thread(target=load_model, args=(kind, model_id, args.device), daemon=True).start()
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    print(f"Tide Pool on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
