"""Embed and analyse EVERY patch (378,680) for the full-dataset viewer at /full/.

    python pipeline/full.py                                   # all patches, EmbeddingGemma 2
    python pipeline/full.py --limit 20000 --embedder stub     # quick try-out on a random subset

Unlike build.py (a few thousand patches, everything in the browser), this is built to scale:

  1. index    every crop + its survey metadata         -> data/full/patches.parquet
  2. embed    shards of 4,096 patches, resumable        -> data/full/emb_<model>/shard_00000.npy
              crops are streamed from the bucket, embedded and dropped (no 32 GB copy on disk)
  3. analyse  k-NN on the GPU when available, zero-shot, k-means, PCA, UMAP
                                                        -> data/full/analysis_<model>.npz
  4. write    compact binary columns for the WebGL viewer -> web/full/data/
              and the search matrix the server uses     -> data/full/search_<model>.npy

Stop it at any time and run the same command again: finished shards are skipped, and the
analysis is reused unless you pass --reanalyse. Free-text search, image search and
"find similar" run in the server (server/app.py) over all vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import build  # noqa: E402
import embedders  # noqa: E402
import gcs  # noqa: E402
import labels as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FULL = ROOT / "data" / "full"
OUT = ROOT / "web" / "full" / "data"
SHARD = 4096
log = build.log


# --------------------------------------------------------------------------- 1. index
def load_patches(args) -> pd.DataFrame:
    """The fixed row order every later stage refers to. Rebuilt only if the selection changes."""
    path, cfg_path = FULL / "patches.parquet", FULL / "patches.json"
    cfg = {"splits": args.splits, "limit": args.limit, "seed": args.seed}
    if path.exists() and cfg_path.exists():
        if json.loads(cfg_path.read_text()) == cfg:
            return pd.read_parquet(path)
        sys.exit(f"{FULL} was built for {cfg_path.read_text()}, not {cfg}. "
                 "Delete data/full/ (or pass the same --limit/--splits/--seed) to continue.")
    index = build.load_index(refresh=False)
    if args.limit:
        s = build.sample(index, args.limit, "proportional", args.splits.split(","), args.seed)
    else:
        s = index[index.split.isin(args.splits.split(","))].sort_values("object").reset_index(drop=True)
        log(f"using all {len(s):,} patches in splits {args.splits}")
    s = build.join_metadata(s)
    keep = ["object", "file", "split", "label", "site", "island_code", "year", "image_name", "point",
            "source", "Region", "Island", "Date", "Latitude", "Longitude", "Row", "Column", "cn_pred", "cn_conf"]
    keep += [f"Machine {k} {i}" for i in range(1, 6) for k in ("suggestion", "confidence")]
    s = s[[c for c in keep if c in s]].reset_index(drop=True)
    for c in s.columns:
        if s[c].dtype == object:
            s[c] = s[c].astype("string")
    FULL.mkdir(parents=True, exist_ok=True)
    s.to_parquet(path)
    cfg_path.write_text(json.dumps(cfg))
    return s


# --------------------------------------------------------------------------- 2. embed
def fetch(objects: list[str], workers: int) -> list[Image.Image | None]:
    def one(o):
        try:
            return Image.open(io.BytesIO(gcs._get(gcs.public_url(o)).content)).convert("RGB")
        except Exception as e:  # a missing crop becomes an empty row, flagged in `ok`
            log(f"  ! {o}: {e}")
            return None

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, objects))


def save_npy(path: Path, a: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        np.save(f, a)
    tmp.replace(path)  # atomic: an interrupted run never leaves a half-written shard


def embed_all(s: pd.DataFrame, emb, slug: str, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (embeddings float16 n x d, ok mask, mean rgb uint8 n x 3)."""
    d = FULL / f"emb_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    shards = [(k, range(b, min(b + SHARD, len(s)))) for k, b in enumerate(range(0, len(s), SHARD))]
    todo = [(k, r) for k, r in shards if not (d / f"shard_{k:05d}.npy").exists()]
    log(f"embedding {len(s):,} patches in {len(shards)} shards with {emb.name} "
        f"({len(shards) - len(todo)} shards already done)")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(1) as pre:  # downloads shard k+1 while shard k is on the GPU
        nxt = pre.submit(fetch, list(s.object.iloc[todo[0][1]]), args.workers) if todo else None
        for j, (k, r) in enumerate(todo):
            imgs = nxt.result()
            if j + 1 < len(todo):
                nxt = pre.submit(fetch, list(s.object.iloc[todo[j + 1][1]]), args.workers)
            ok = np.array([im is not None for im in imgs])
            vecs, rgb = None, np.zeros((len(imgs), 3), np.uint8)
            good = [im for im in imgs if im is not None]
            for b in range(0, len(good), args.batch):
                v = emb.images(good[b : b + args.batch])
                vecs = np.zeros((len(imgs), v.shape[1]), np.float16) if vecs is None else vecs
                vecs[np.flatnonzero(ok)[b : b + len(v)]] = v.astype(np.float16)
            for i, im in enumerate(imgs):
                if im is not None:
                    rgb[i] = np.asarray(im.resize((8, 8)), np.float32).reshape(-1, 3).mean(0)
            save_npy(d / f"rgb_{k:05d}.npy", rgb)
            save_npy(d / f"ok_{k:05d}.npy", ok)
            save_npy(d / f"shard_{k:05d}.npy", vecs)  # last: its existence marks the shard done
            done += len(r)
            rate = done / (time.time() - t0)
            left = sum(len(rr) for _, rr in todo[j + 1 :])
            log(f"  shard {k + 1}/{len(shards)}  {rate:.0f} patches/s  ~{left / max(rate, 1e-9) / 60:.0f} min left"
                + (f"  ({(~ok).sum()} failed downloads)" if (~ok).any() else ""))
    load = lambda p: np.concatenate([np.load(d / f"{p}_{k:05d}.npy") for k, _ in shards])
    return load("shard"), load("ok"), load("rgb")


# --------------------------------------------------------------------------- 3. analyse
def knn(x: np.ndarray, ok: np.ndarray, k: int, device: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine k-NN. On CUDA, 378k x 768 takes a few minutes; on CPU it is much slower."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is None:
        log("  torch not installed: k-NN on CPU with numpy")
        xx = x.copy()
        xx[~ok] = 0
        idx, sim = build.knn(xx, k)
        return idx, sim
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dev == "cuda" else torch.float32
    log(f"  k-NN on {dev}")
    xt = torch.from_numpy(x).to(dev, dt)
    bad = torch.from_numpy(~ok).to(dev)
    n, B = len(x), 1024  # a 1024 x 378k float32 block is ~1.6 GB: fits next to the data on most GPUs
    idx = np.empty((n, k), np.int32)
    sim = np.empty((n, k), np.float32)
    t0 = time.time()
    for b in range(0, n, B):
        e = min(b + B, n)
        s = (xt[b:e] @ xt.T).float()
        s[:, bad] = -2  # never pick a patch whose crop failed to download
        s[torch.arange(e - b), torch.arange(b, e)] = -2  # nor the patch itself
        v, i = torch.topk(s, k, dim=1)
        idx[b:e], sim[b:e] = i.cpu().numpy(), v.cpu().numpy()
        if (b // B) % 100 == 0:
            log(f"    {e:,}/{n:,}  ({time.time() - t0:.0f}s)")
    return idx, sim


def analyse(s, X, ok, emb, slug, codes, args) -> dict:
    path = FULL / f"analysis_{slug}.npz"
    if path.exists() and not args.reanalyse:
        log(f"reusing {path.name} (pass --reanalyse to recompute)")
        return dict(np.load(path))
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    x = X[:, : args.dim].astype(np.float32)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    x[~ok] = 0
    y = np.array([codes.index(c) for c in s.label], np.int16)
    out: dict[str, np.ndarray] = {}

    log("k-nearest neighbours ...")
    nn, nn_sim = knn(x, ok, args.k, args.device)
    out["nn"], out["nn_sim"] = nn, nn_sim.astype(np.float16)
    counts = np.zeros((len(x), len(codes)), np.int16)
    np.add.at(counts, (np.repeat(np.arange(len(x)), args.k), y[nn].ravel()), 1)
    out["knn_pred"] = counts.argmax(1).astype(np.int16)
    out["knn_agree"] = (y[nn] == y[:, None]).mean(1).astype(np.float32)
    out["atypicality"] = (1 - nn_sim.mean(1)).astype(np.float32)

    log("zero-shot text prototypes ...")
    t3 = build.matryoshka(emb.texts([L.PROMPT_TEMPLATE.format(name=L.prompt_of(c)) for c in codes]), args.dim)
    zs = x @ t3.T.astype(np.float32)
    top = np.argsort(-zs, axis=1)[:, :3]
    out["zs_top"] = top.astype(np.int16)
    out["zs_score"] = np.take_along_axis(zs, top, 1).astype(np.float16)

    log("class centroids ...")
    cent = np.stack([x[(y == c) & ok].mean(0) if ((y == c) & ok).any() else np.zeros(x.shape[1]) for c in range(len(codes))])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)
    out["centroid_sim"] = (x * cent[y]).sum(1).astype(np.float32)

    kc = args.clusters
    log(f"k-means ({kc}) ...")
    km = MiniBatchKMeans(kc, batch_size=8192, n_init=3, random_state=args.seed).fit(x[ok])
    out["cluster"] = km.predict(x).astype(np.int16)

    log("PCA ...")
    p50 = PCA(50, svd_solver="randomized", random_state=args.seed).fit_transform(x)
    out["pca"] = p50[:, :2].astype(np.float32)
    if not args.skip_umap:
        import umap

        log(f"UMAP on {len(x):,} points (PCA-50 input; several minutes for the full set) ...")
        out["umap"] = umap.UMAP(n_neighbors=15, min_dist=0.1, low_memory=True,
                                n_jobs=-1).fit_transform(p50).astype(np.float32)
    np.savez(path, **out)
    return out


# --------------------------------------------------------------------------- 4. write
def norm2(p: np.ndarray) -> np.ndarray:
    """Scale to [-1, 1] keeping aspect, robust to outliers (1st-99th percentile)."""
    p = p.astype(np.float32)
    fin = np.isfinite(p).all(1)
    lo, hi = np.percentile(p[fin], 0.5, axis=0), np.percentile(p[fin], 99.5, axis=0)
    c, r = (lo + hi) / 2, max((hi - lo).max() / 2, 1e-9)
    return ((p - c) / r).astype(np.float32)


def codes_of(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    vals = series.astype("string")
    cats = sorted(v for v in vals.dropna().unique())
    lut = {v: i for i, v in enumerate(cats)}
    return np.array([lut.get(v, 255) if pd.notna(v) else 255 for v in vals], np.uint8), cats


def write_viewer(s, X, ok, rgb, A, emb, slug, codes, args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)  # may be a mount point: empty it, don't delete it
    for old in OUT.iterdir():
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    n = len(s)
    info = {c: L.info(c) for c in codes}
    groups = [g for g in L.T1_GROUPS if any(info[c]["t1"] == g for c in codes)]
    t2s = sorted({(info[c]["t2"], info[c]["t2_name"], groups.index(info[c]["t1"])) for c in codes}, key=lambda t: (t[2], t[0]))
    t2bs = sorted({(info[c]["t2b"], info[c]["t2b_name"], groups.index(info[c]["t1"])) for c in codes}, key=lambda t: (t[2], t[0]))
    t2i = {t[0]: i for i, t in enumerate(t2s)}
    t2bi = {t[0]: i for i, t in enumerate(t2bs)}
    y = np.array([codes.index(c) for c in s.label], np.uint8)
    cls_t1 = np.array([groups.index(info[c]["t1"]) for c in codes], np.uint8)
    cls_t2 = np.array([t2i[info[c]["t2"]] for c in codes], np.uint8)
    cls_t2b = np.array([t2bi[info[c]["t2b"]] for c in codes], np.uint8)

    cols: dict[str, tuple[np.ndarray, str]] = {}

    def col(name, a, kind):
        cols[name] = (np.ascontiguousarray(a), kind)

    views = {"umap": A.get("umap"), "pca": A["pca"]}
    lat = pd.to_numeric(s.Latitude, errors="coerce").to_numpy(np.float32)
    lon = pd.to_numeric(s.Longitude, errors="coerce").to_numpy(np.float32)
    if np.isfinite(lat).any():
        lon360 = np.where(lon < 0, lon + 360, lon)  # Pacific spans the antimeridian
        jit = np.random.default_rng(args.seed).normal(0, 0.01, (n, 2)).astype(np.float32)
        geo = np.stack([lon360, lat], 1) + jit
        views["geo"] = np.where(np.isfinite(geo), geo, np.nan)
    for v, p in views.items():
        if p is not None:
            q = norm2(p)
            col(f"xy_{v}", np.nan_to_num(q, nan=9.0).ravel(), "f32")  # 9 = off-screen
    col("label", y, "u8")
    split, splits = codes_of(s.split)
    region, regions = codes_of(s.Region)
    island, islands = codes_of(s.Island)
    col("split", split, "u8")
    col("region", region, "u8")
    col("island", island, "u8")
    col("year", pd.to_numeric(s.year, errors="coerce").fillna(0).to_numpy(np.uint16), "u16")
    col("ok", ok.astype(np.uint8), "u8")
    # bucket paths rebuild as <T3_PREFIX>/<split>/<code>/<code>_<photo stem>_<point>.png, so the viewer
    # gets a photo list and two small columns instead of 378k path strings
    stems, photo = np.unique(s.image_name.str.slice(0, -4).to_numpy(str), return_inverse=True)
    rebuilt = [f"{gcs.T3_PREFIX}/{sp}/{lb}/{lb}_{stems[p]}_{pt}.png" for sp, lb, p, pt in zip(s.split, s.label, photo, s.point)]
    if (bad := int((np.array(rebuilt) != s.object.to_numpy(str)).sum())):
        log(f"  ! {bad} patch paths do not follow the naming pattern; their images will not load")
    source, sources = codes_of(s.source)
    col("photo", photo.astype(np.uint32), "u32")
    col("point", s.point.to_numpy(np.uint16), "u16")
    col("source", source, "u8")
    (OUT / "photos.json").write_text(json.dumps({"t3_prefix": gcs.T3_PREFIX, "sources": sources,
                                                 "photos": stems.tolist()}, separators=(",", ":")))
    col("rgb", rgb.ravel(), "u8")
    col("cluster", A["cluster"].astype(np.uint8), "u8")
    col("knn_agree", A["knn_agree"], "f32")
    col("knn_correct", (A["knn_pred"] == y).astype(np.uint8), "u8")
    col("zs_pred", A["zs_top"][:, 0].astype(np.uint8), "u8")
    col("atypicality", A["atypicality"], "f32")
    col("centroid_sim", A["centroid_sim"], "f32")
    cn = s.cn_pred.astype("string")
    cn_idx = np.array([codes.index(c) if pd.notna(c) and c in codes else 255 for c in cn], np.uint8)
    col("cn_pred", cn_idx, "u8")
    col("cn_conf", pd.to_numeric(s.cn_conf, errors="coerce").to_numpy(np.float32), "f32")

    files = {}
    for name, (a, kind) in cols.items():
        a.astype({"f32": "<f4", "u8": "u1", "u16": "<u2", "u32": "<u4"}[kind]).tofile(OUT / f"{name}.bin")
        files[name] = {"file": f"{name}.bin", "type": kind}

    # per-shard detail records, fetched by the viewer only when a patch is opened
    (OUT / "detail").mkdir()
    for k, b in enumerate(range(0, n, SHARD)):
        e = min(b + SHARD, n)
        part = s.iloc[b:e]
        machine = []
        for _, r in part.iterrows():
            sug = []
            for i in range(1, 6):
                c, v = r.get(f"Machine suggestion {i}"), r.get(f"Machine confidence {i}")
                if pd.notna(c):
                    sug.append([str(c).lstrip("*"), None if pd.isna(v) else float(v)])
            machine.append(sug)
        num = lambda c: [None if pd.isna(v) else round(float(v), 5) for v in pd.to_numeric(part[c], errors="coerce")]
        rec = {
            "site": part.site.tolist(),
            "date": [None if pd.isna(v) else str(v) for v in part.Date] if "Date" in part else [None] * len(part),
            "lat": num("Latitude"), "lon": num("Longitude"), "row": num("Row"), "col": num("Column"),
            "nn": A["nn"][b:e].tolist(), "nn_sim": np.round(A["nn_sim"][b:e].astype(np.float32), 3).tolist(),
            "zs_top": A["zs_top"][b:e].tolist(), "zs_score": np.round(A["zs_score"][b:e].astype(np.float32), 4).tolist(),
            "machine": machine,
        }
        (OUT / "detail" / f"{k:05d}.json").write_text(json.dumps(rec, separators=(",", ":")))

    # search matrix for the server: web-dim prefix, re-normalised, float16
    w = X[:, : args.web_dim].astype(np.float32)
    w /= np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9)
    w[~ok] = 0
    save_npy(FULL / "search.npy", w.astype(np.float16))

    y_ok = ok
    meta = {
        "title": "NOAA Pacific benthic cover (Tier 3), all patches",
        "dataset_url": "https://huggingface.co/datasets/NMFS-OSI/noaa-pacific-benthic-cover-t3",
        "bucket": gcs.BUCKET, "mirror": gcs.MIRROR,
        "model": emb.name, "model_is_stub": args.embedder == "stub", "model_id": args.model,
        "dim": args.dim, "web_dim": args.web_dim, "k": args.k, "shard": SHARD,
        "n": n, "n_ok": int(ok.sum()), "built": time.strftime("%Y-%m-%d %H:%M"),
        "columns": files, "views": [v for v in ("umap", "pca", "geo") if f"xy_{v}" in cols],
        "groups": [{"code": g, "name": L.T1_GROUPS[g][0]} for g in groups],
        "classes": [{"code": c, "name": info[c]["name"], "group": int(cls_t1[i]), "t2": int(cls_t2[i]),
                     "t2b": int(cls_t2b[i]), "n": int((y == i).sum())} for i, c in enumerate(codes)],
        "t2": [{"code": t[0], "name": t[1], "group": t[2]} for t in t2s],
        "t2b": [{"code": t[0], "name": t[1], "group": t[2]} for t in t2bs],
        "splits": splits, "regions": regions, "islands": islands,
        "clusters": int(A["cluster"].max()) + 1,
        "metrics": {
            "knn_acc": float((A["knn_pred"] == y)[y_ok].mean()),
            "knn_agree": float(A["knn_agree"][y_ok].mean()),
            "zero_shot_acc": float((A["zs_top"][:, 0] == y)[y_ok].mean()),
            "coralnet_agree": float((cn_idx == y)[cn_idx != 255].mean()) if (cn_idx != 255).any() else None,
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e6
    log(f"wrote {OUT} ({size:.0f} MB) and {FULL / 'search.npy'}")
    log("metrics: " + json.dumps({k: (round(v, 3) if v is not None else None) for k, v in meta["metrics"].items()}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="random subset of this size (0 = every patch)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--embedder", choices=["gemma", "stub"], default="gemma")
    ap.add_argument("--model", default=embedders.MODEL_ID)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=32, help="parallel crop downloads")
    ap.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128], help="dimension for analysis")
    ap.add_argument("--web-dim", type=int, default=256, choices=[768, 512, 256, 128], help="dimension for search")
    ap.add_argument("--k", type=int, default=10, help="neighbours per patch")
    ap.add_argument("--clusters", type=int, default=64, help="k-means clusters")
    ap.add_argument("--skip-umap", action="store_true", help="PCA and map views only")
    ap.add_argument("--reanalyse", action="store_true", help="recompute k-NN / clusters / UMAP")
    args = ap.parse_args()

    s = load_patches(args)
    emb = embedders.make(args.embedder, args.model, args.device, args.batch)
    slug = hashlib.md5(emb.name.encode()).hexdigest()[:8]
    X, ok, rgb = embed_all(s, emb, slug, args)
    codes = sorted(s.label.unique(), key=lambda c: (list(L.T1_GROUPS).index(L.group_of(c))
                                                    if L.group_of(c) in L.T1_GROUPS else 99, c))
    A = analyse(s, X, ok, emb, slug, codes, args)
    write_viewer(s, X, ok, rgb, A, emb, slug, codes, args)


if __name__ == "__main__":
    main()
