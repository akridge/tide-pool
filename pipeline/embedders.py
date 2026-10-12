"""Embedding backends shared by the build pipeline and the search server."""
from __future__ import annotations

import hashlib
import time

import numpy as np
from PIL import Image

MODEL_ID = "google/embeddinggemma-2"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class GemmaEmbedder:
    name = MODEL_ID

    def __init__(self, model_id: str, device: str | None, batch: int):
        import torch
        from sentence_transformers import SentenceTransformer

        dev = device or ("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
        # float16 overflows this model's activations; bf16 on GPU, fp32 elsewhere
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        log(f"loading {model_id} on {dev} ({dtype}) without the audio encoder ...")
        try:
            self.model = SentenceTransformer(
                model_id, device=dev,
                config_kwargs={"audio_config": None},
                model_kwargs={"torch_dtype": dtype},
            )
        except Exception as e:  # older/newer loaders may reject dropping an encoder
            log(f"could not skip the audio encoder ({e}); loading the full model")
            self.model = SentenceTransformer(model_id, device=dev, model_kwargs={"torch_dtype": dtype})
        self.name, self.batch = model_id, batch
        self.prompts = dict(getattr(self.model, "prompts", {}) or {})
        log(f"model prompts: {list(self.prompts)}")

    def _prompt(self, *candidates: str) -> str | None:
        return next((p for p in candidates if p in self.prompts), None)

    def images(self, imgs: list[Image.Image]) -> np.ndarray:
        return self.model.encode([{"image": im} for im in imgs], batch_size=self.batch,
                                 convert_to_numpy=True, normalize_embeddings=True)

    def texts(self, texts: list[str], kind: str = "document") -> np.ndarray:
        p = (self._prompt("Document", "document") if kind == "document"
             else self._prompt("SearchQuery", "query", "Query"))
        return self.model.encode(texts, prompt_name=p, batch_size=self.batch,
                                 convert_to_numpy=True, normalize_embeddings=True)


class StubEmbedder:
    """Deterministic colour/texture features in a random 768-d basis. NOT a semantic model;
    exists only so the pipeline and UI can be exercised without downloading the model."""

    name = "stub (colour-texture features, not EmbeddingGemma)"
    prompts: dict = {}

    def __init__(self):
        self.P = np.random.default_rng(0).standard_normal((96, 768)).astype(np.float32)

    def _feat(self, im: Image.Image) -> np.ndarray:
        a = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32) / 255
        hist = np.concatenate([np.histogram(a[..., c], 16, (0, 1))[0] for c in range(3)]) / 1024
        g = a.mean(-1)
        grad = np.concatenate([np.histogram(np.abs(np.diff(g, axis=ax)), 16, (0, .5))[0]
                               for ax in (0, 1)]) / 1000
        f = np.concatenate([hist, grad, a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0),
                            np.zeros(96 - 48 - 32 - 6)])
        v = (f - 0.05) @ self.P
        return v / (np.linalg.norm(v) + 1e-9)

    def images(self, imgs):
        return np.stack([self._feat(im) for im in imgs])

    def texts(self, texts, kind="document"):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(768)
            out.append(v / np.linalg.norm(v))
        return np.stack(out).astype(np.float32)


def make(kind: str, model_id: str = MODEL_ID, device: str | None = None, batch: int = 32):
    return StubEmbedder() if kind == "stub" else GemmaEmbedder(model_id, device, batch)
