"""Build the Tide Pool explorer bundle.

Samples benthic patches from the public NOAA PIFSC CoralNet mirror (the files behind
the Hugging Face dataset NMFS-OSI/noaa-pacific-benthic-cover-t3), joins each patch to
its survey metadata, embeds the patches with google/embeddinggemma-2, and writes
everything the web app needs into web/data/.

    python pipeline/build.py --n 6000
    python pipeline/build.py --n 300 --embedder stub     # quick UI smoke test, no model

Every stage caches under data/cache/, so re-running with a bigger --n only downloads
and embeds the new patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import embedders  # noqa: E402
import export  # noqa: E402
import gcs  # noqa: E402
import labels as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "web" / "data"
MODEL_ID = embedders.MODEL_ID
THUMB = 64
SHEET_COLS = 32  # 32x32 thumbs per 2048px sprite sheet


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- index
def parse_name(obj: str) -> dict | None:
    """.../pacific_t3_splits/<split>/<LABEL>/<LABEL>_<SITE>_<YEAR>_<...>_<PHOTO>_<POINT>.png"""
    parts = obj.split("/")
    if not obj.endswith(".png") or len(parts) < 3:
        return None
    split, label, base = parts[-3], parts[-2], parts[-1][:-4]
    if not base.startswith(label + "_"):
        return None
    stem, _, point = base[len(label) + 1 :].rpartition("_")
    toks = stem.split("_")
    site = toks[0]
    year = next((int(t) for t in toks[1:] if t.isdigit() and len(t) == 4), None)
    return {
        "object": obj,
        "split": split,
        "label": label,
        "file": base + ".png",
        "image_name": stem + ".JPG",
        "point": int(point) if point.isdigit() else -1,
        "site": site,
        "island_code": site.split("-")[0],
        "year": year,
    }


def load_index(refresh: bool) -> pd.DataFrame:
    path = CACHE / "t3_index.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    log("listing gs://%s/%s ..." % (gcs.BUCKET, gcs.T3_PREFIX))
    rows = [r for name, _ in gcs.list_objects(gcs.T3_PREFIX + "/") if (r := parse_name(name))]
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log(f"indexed {len(df):,} patches, {df.label.nunique()} classes")
    return df


def sample(df: pd.DataFrame, n: int, strategy: str, splits: list[str], seed: int) -> pd.DataFrame:
    df = df[df.split.isin(splits)]
    rng = np.random.default_rng(seed)
    counts = df.label.value_counts()
    if strategy == "proportional":
        take = (counts / counts.sum() * n).round().clip(lower=1).astype(int)
    else:  # balanced: equal share per class, leftover share redistributed to bigger classes
        take, budget, remaining = {}, n, counts.sort_values()
        for i, (lab, c) in enumerate(remaining.items()):
            share = budget // (len(remaining) - i)
            take[lab] = min(c, share)
            budget -= take[lab]
        take = pd.Series(take)
    parts = []
    for lab, k in take.items():
        g = df[df.label == lab]
        parts.append(g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)])
    out = pd.concat(parts).sort_values("object").reset_index(drop=True)
    log(f"sampled {len(out):,} patches ({strategy}) from {len(df):,} in splits {splits}")
    return out


# --------------------------------------------------------------------------- metadata
META_COLS = ["Name", "Date", "Region", "Island", "Site", "Latitude", "Longitude", "Depth",
             "Row", "Column", "Label code"] + [f"Machine {k} {i}" for i in range(1, 6)
                                               for k in ("suggestion", "confidence")]


def load_annotations() -> pd.DataFrame:
    path = CACHE / "annotations.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for source, fname in gcs.SOURCES.items():
        local = CACHE / "csv" / fname
        log(f"downloading {source} annotations ...")
        gcs.download(f"{gcs.MIRROR}/{source}/{fname}", local)
        d = pd.read_csv(local, usecols=lambda c: c in META_COLS, low_memory=False)
        d["point"] = d.groupby("Name").cumcount() + 1  # crop suffix is the 1-based point order
        d["source"] = source
        frames.append(d)
    ann = pd.concat(frames, ignore_index=True)
    for c in ann.columns:
        if ann[c].dtype == object:
            ann[c] = ann[c].astype("string")
    ann.to_parquet(path)
    return ann


def join_metadata(s: pd.DataFrame) -> pd.DataFrame:
    ann = load_annotations()
    m = s.merge(ann, how="left", left_on=["image_name", "point"], right_on=["Name", "point"])
    code = m["Label code"].astype("string").str.lstrip("*")
    m["meta_match"] = (code == m["label"]).fillna(False)
    log(f"metadata joined for {m.Name.notna().mean():.1%} of patches; "
        f"label agrees with annotation CSV for {m.meta_match.mean():.1%}")
    # CoralNet machine suggestion (the CoralNet classifier's top-1 at annotation time)
    m["cn_pred"] = m["Machine suggestion 1"].astype("string").str.lstrip("*")
    m["cn_conf"] = pd.to_numeric(m["Machine confidence 1"], errors="coerce")
    return m


# --------------------------------------------------------------------------- images
def fetch_crops(s: pd.DataFrame, workers: int) -> list[Path | None]:
    pairs = [(o, CACHE / "crops" / o.split("/")[-2] / f) for o, f in zip(s.object, s.file)]
    missing = sum(not p.exists() for _, p in pairs)
    log(f"downloading {missing:,} crops ({len(pairs) - missing:,} cached) ...")
    return gcs.download_many(pairs, workers)


def image_stats(img: Image.Image) -> dict:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    gray = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:])
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    hist = np.histogram(gray, bins=64, range=(0, 1))[0] / gray.size
    ang = hsv[..., 0] * 2 * np.pi
    w = hsv[..., 1] + 1e-6  # saturation-weighted circular mean hue
    hue = (np.degrees(np.arctan2((np.sin(ang) * w).sum(), (np.cos(ang) * w).sum())) + 360) % 360
    mean = a.reshape(-1, 3).mean(0)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv[..., 1].mean()),
        "hue": float(hue),
        "colorfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
        "sharpness": float(np.log1p(lap.var() * 1e4)),
        "entropy": float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()),
        "rgb": "#%02x%02x%02x" % tuple(int(v * 255) for v in mean),
    }


def embed_images(emb, s: pd.DataFrame, paths: list[Path], batch: int) -> np.ndarray:
    slug = hashlib.md5(emb.name.encode()).hexdigest()[:8]
    cache_path = CACHE / f"emb_{slug}.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    todo = [i for i, f in enumerate(s.file) if f not in cache]
    log(f"embedding {len(todo):,} patches with {emb.name} ({len(s) - len(todo):,} cached) ...")
    t0 = time.time()
    for b in range(0, len(todo), batch):
        idx = todo[b : b + batch]
        imgs = [Image.open(paths[i]).convert("RGB") for i in idx]
        vecs = emb.images(imgs)
        for i, v in zip(idx, vecs):
            cache[s.file.iloc[i]] = v.astype(np.float16)
        done = b + len(idx)
        if done % (batch * 20) < batch or done == len(todo):
            rate = done / (time.time() - t0)
            log(f"  {done:,}/{len(todo):,}  ({rate:.1f} img/s)")
            np.savez(cache_path, **cache)
    return np.stack([cache[f].astype(np.float32) for f in s.file])


def matryoshka(x: np.ndarray, dim: int) -> np.ndarray:
    x = x[:, :dim]
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


# --------------------------------------------------------------------------- analysis
def projections(x: np.ndarray, seed: int, skip_tsne: bool) -> dict[str, np.ndarray]:
    from sklearn.decomposition import PCA

    out = {}
    log("PCA ...")
    out["pca"] = PCA(2, random_state=seed).fit_transform(x)
    try:
        import umap

        nn = min(15, len(x) - 1)
        log("UMAP 2D ...")
        out["umap"] = umap.UMAP(n_neighbors=nn, min_dist=0.1, metric="cosine",
                                random_state=seed).fit_transform(x)
        log("UMAP 3D ...")
        out["umap3d"] = umap.UMAP(n_components=3, n_neighbors=nn, min_dist=0.1, metric="cosine",
                                  random_state=seed).fit_transform(x)
    except ImportError:
        log("umap-learn not installed; skipping UMAP")
    if not skip_tsne:
        from sklearn.manifold import TSNE

        log("t-SNE ...")
        out["tsne"] = TSNE(2, metric="cosine", init="pca", random_state=seed,
                           perplexity=min(30, (len(x) - 1) / 3)).fit_transform(x)
    return out


def norm_coords(p: np.ndarray) -> np.ndarray:
    p = p - np.median(p, axis=0)
    scale = np.percentile(np.abs(p), 99.5) or 1
    return np.clip(p / scale, -1.2, 1.2)


def knn(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sims_all = np.empty((len(x), k), np.float32)
    idx_all = np.empty((len(x), k), np.int32)
    for b in range(0, len(x), 2048):
        s = x[b : b + 2048] @ x.T
        for j in range(s.shape[0]):
            s[j, b + j] = -np.inf
        idx = np.argpartition(-s, k, axis=1)[:, :k]
        part = np.take_along_axis(s, idx, 1)
        order = np.argsort(-part, axis=1)
        idx_all[b : b + 2048] = np.take_along_axis(idx, order, 1)
        sims_all[b : b + 2048] = np.take_along_axis(part, order, 1)
    return idx_all, sims_all


def linear_probe(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    """Out-of-fold logistic-regression probabilities, grouped by source photo so
    patches from the same photo never sit on both sides of a split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    classes = np.unique(y)
    proba = np.zeros((len(x), len(classes)), np.float32)
    min_count = Counter(y).most_common()[-1][1]
    folds = max(2, min(5, min_count))
    cv = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f, (tr, te) in enumerate(cv.split(x, y, groups)):
            clf = LogisticRegression(max_iter=2000, C=2.0)
            clf.fit(x[tr], y[tr])
            proba[np.ix_(te, np.searchsorted(classes, clf.classes_))] = clf.predict_proba(x[te])
            log(f"  probe fold {f + 1}/{folds}")
    return classes, proba


def topk(scores: np.ndarray, k: int):
    idx = np.argsort(-scores, axis=1)[:, :k]
    return idx, np.take_along_axis(scores, idx, 1)


# --------------------------------------------------------------------------- export
def write_sprites(paths: list[Path], out: Path) -> int:
    per = SHEET_COLS * SHEET_COLS
    sheets = math.ceil(len(paths) / per)
    for sh in range(sheets):
        canvas = Image.new("RGB", (SHEET_COLS * THUMB, SHEET_COLS * THUMB), (20, 20, 20))
        for j, p in enumerate(paths[sh * per : (sh + 1) * per]):
            im = Image.open(p).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
            canvas.paste(im, ((j % SHEET_COLS) * THUMB, (j // SHEET_COLS) * THUMB))
        canvas.save(out / f"sprite_{sh}.jpg", quality=82)
    return sheets


def rnd(a, d=4):
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), d)
            for v in a]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000, help="number of patches to sample")
    ap.add_argument("--strategy", choices=["balanced", "proportional"], default="balanced")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128],
                    help="Matryoshka dimension used for analysis/projections")
    ap.add_argument("--web-dim", type=int, default=256, choices=[768, 512, 256, 128],
                    help="Matryoshka dimension shipped to the browser for similarity search")
    ap.add_argument("--embedder", choices=["gemma", "stub"], default="gemma")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--k", type=int, default=12, help="neighbours stored per patch")
    ap.add_argument("--clusters", type=int, default=0, help="k-means clusters (0 = auto)")
    ap.add_argument("--skip-tsne", action="store_true")
    ap.add_argument("--copy-crops", action="store_true",
                    help="copy full-size crops into web/data (offline use) instead of loading from GCS")
    ap.add_argument("--labels-csv", help="CSV (code,name[,group]) overriding class names")
    ap.add_argument("--refresh-index", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--export", metavar="DIR",
                    help="also write metadata.parquet + embeddings.zarr (full dimension) to DIR")
    args = ap.parse_args()

    if args.labels_csv:
        L.load_overrides(args.labels_csv)

    # 1. sample + metadata
    index = load_index(args.refresh_index)
    s = sample(index, args.n, args.strategy, args.splits.split(","), args.seed)
    s = join_metadata(s)
    paths = fetch_crops(s, args.workers)
    ok = np.array([p is not None for p in paths])
    s, paths = s[ok].reset_index(drop=True), [p for p in paths if p is not None]

    # 2. per-image stats
    log("image statistics ...")
    stats = pd.DataFrame([image_stats(Image.open(p)) for p in paths])

    # 3. embeddings
    emb = embedders.make(args.embedder, args.model, args.device, args.batch)
    full = embed_images(emb, s, paths, args.batch)
    x = matryoshka(full, args.dim)

    codes = sorted(s.label.unique(), key=lambda c: (list(L.T1_GROUPS).index(L.group_of(c))
                                                    if L.group_of(c) in L.T1_GROUPS else 99, c))
    group_codes = [g for g in L.T1_GROUPS if any(L.group_of(c) == g for c in codes)]
    y = np.array([codes.index(c) for c in s.label])
    yg = np.array([group_codes.index(L.group_of(c)) for c in s.label])

    # 4. zero-shot against text descriptions in the shared embedding space
    log("zero-shot text prototypes ...")
    t3_txt = matryoshka(emb.texts([L.PROMPT_TEMPLATE.format(name=L.prompt_of(c)) for c in codes]), args.dim)
    t1_txt = matryoshka(emb.texts([L.PROMPT_TEMPLATE.format(name=L.T1_GROUPS[g][1]) for g in group_codes]),
                        args.dim)
    zs = x @ t3_txt.T
    zs_g = x @ t1_txt.T
    zs_idx, zs_val = topk(zs, 5)

    # 5. neighbourhoods
    log("k-nearest neighbours ...")
    nn_idx, nn_sim = knn(x, args.k)
    agree = (y[nn_idx] == y[:, None]).mean(1)
    agree_g = (yg[nn_idx] == yg[:, None]).mean(1)
    knn_pred = np.array([np.bincount(r, minlength=len(codes)).argmax() for r in y[nn_idx]])
    atyp = 1 - nn_sim.mean(1)
    centroids = np.stack([x[y == c].mean(0) for c in range(len(codes))])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    cent_sim = (x * centroids[y]).sum(1)

    # 6. linear probe (how linearly separable are the classes in this space?)
    log("linear probe (grouped cross-validation) ...")
    _, proba = linear_probe(x, y, s.image_name.values, args.seed)
    lp_idx, lp_val = topk(proba, 5)
    proba_g = np.zeros((len(x), len(group_codes)), np.float32)
    for c, code in enumerate(codes):
        proba_g[:, group_codes.index(L.group_of(code))] += proba[:, c]

    # 7. clusters
    from sklearn.cluster import KMeans

    k_clusters = args.clusters or int(np.clip(round(math.sqrt(len(x) / 4)), 8, 40))
    log(f"k-means ({k_clusters}) ...")
    km = KMeans(k_clusters, n_init=4, random_state=args.seed).fit(x)
    cluster = km.labels_

    # 8. projections
    proj = projections(x, args.seed, args.skip_tsne)
    lon = s.Longitude.astype(float).to_numpy()
    lat = s.Latitude.astype(float).to_numpy()
    if np.isfinite(lat).any():
        lon360 = np.where(lon < 0, lon + 360, lon)  # Pacific spans the antimeridian
        rng = np.random.default_rng(args.seed)
        geo = np.stack([lon360, lat], 1) + rng.normal(0, 0.02, (len(x), 2))
        geo[~np.isfinite(geo).all(1)] = np.nan
        proj["geo"] = geo

    # ---------------------------------------------------------------- write bundle
    # empty web/data rather than delete it: it may be a mount point (docker compose)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    log("sprites ...")
    sheets = write_sprites(paths, OUT)
    if args.copy_crops:
        (OUT / "crops").mkdir()
        for i, p in enumerate(paths):
            Image.open(p).convert("RGB").save(OUT / "crops" / f"{i}.jpg", quality=88)

    web = matryoshka(full, args.web_dim).astype(np.float32)
    web.tofile(OUT / "embeddings.f32")
    matryoshka(np.concatenate([t3_txt, t1_txt]), args.web_dim).astype(np.float32).tofile(OUT / "text.f32")

    def coords(name, p):
        if name == "geo":
            lo, la = p[:, 0], p[:, 1]
            cx, cy = np.nanmedian(lo), np.nanmedian(la)
            sc = max(np.nanpercentile(np.abs(lo - cx), 99.5), np.nanpercentile(np.abs(la - cy), 99.5)) or 1
            q = np.stack([(lo - cx) / sc, (la - cy) / sc], 1)
        else:
            q = norm_coords(p)
        return [rnd(q[:, d], 4) for d in range(q.shape[1])]

    def col(series, kind="str"):
        if kind == "str":
            return [None if pd.isna(v) else str(v) for v in series]
        return rnd(pd.to_numeric(series, errors="coerce").to_numpy(float), 4)

    machine = []
    for _, r in s.iterrows():
        sug = []
        for i in range(1, 6):
            code, conf = r.get(f"Machine suggestion {i}"), r.get(f"Machine confidence {i}")
            if pd.notna(code):
                sug.append([str(code).lstrip("*"), None if pd.isna(conf) else float(conf)])
        machine.append(sug)

    points = {
        "file": s.file.tolist(),
        "object": s.object.tolist(),
        "split": s.split.tolist(),
        "label": y.tolist(),
        "group": yg.tolist(),
        "site": s.site.tolist(),
        "island_code": s.island_code.tolist(),
        "year": [None if pd.isna(v) else int(v) for v in s.year],
        "image_name": s.image_name.tolist(),
        "point": s.point.tolist(),
        "source": col(s.source),
        "region": col(s.Region),
        "island": col(s.Island),
        "date": col(s.Date),
        "lat": col(s.Latitude, "num"),
        "lon": col(s.Longitude, "num"),
        "depth": col(s.Depth, "num"),
        "row": [None if pd.isna(v) else int(v) for v in s.Row],
        "col": [None if pd.isna(v) else int(v) for v in s.Column],
        "meta_match": s.meta_match.astype(bool).tolist(),
        "cn_pred": col(s.cn_pred),
        "cn_conf": col(s.cn_conf, "num"),
        "machine": machine,
        "cluster": cluster.tolist(),
        "nn": nn_idx.tolist(),
        "nn_sim": [rnd(r, 3) for r in nn_sim],
        "knn_agree": rnd(agree, 3),
        "knn_agree_group": rnd(agree_g, 3),
        "knn_pred": knn_pred.tolist(),
        "atypicality": rnd(atyp, 4),
        "centroid_sim": rnd(cent_sim, 4),
        "zs_top": zs_idx.tolist(),
        "zs_score": [rnd(r, 4) for r in zs_val],
        "zs_group_pred": zs_g.argmax(1).tolist(),
        "lp_top": lp_idx.tolist(),
        "lp_prob": [rnd(r, 3) for r in lp_val],
        "lp_group_pred": proba_g.argmax(1).tolist(),
        "lp_group_conf": rnd(proba_g.max(1), 3),
        "rgb": stats.rgb.tolist(),
        **{k: rnd(stats[k].to_numpy(), 4) for k in
           ["brightness", "contrast", "saturation", "hue", "colorfulness", "sharpness", "entropy"]},
        "proj": {k: coords(k, v) for k, v in proj.items()},
    }
    # derived correctness flags
    points["lp_correct"] = (lp_idx[:, 0] == y).astype(int).tolist()
    points["zs_correct"] = (zs_idx[:, 0] == y).astype(int).tolist()
    points["knn_correct"] = (knn_pred == y).astype(int).tolist()

    total = index.label.value_counts()
    manifest = {
        "title": "NOAA Pacific benthic cover (Tier 3)",
        "dataset": "NMFS-OSI/noaa-pacific-benthic-cover-t3",
        "dataset_url": "https://huggingface.co/datasets/NMFS-OSI/noaa-pacific-benthic-cover-t3",
        "bucket": gcs.BUCKET,
        "mirror": gcs.MIRROR,
        "model": emb.name,
        "model_is_stub": args.embedder == "stub",
        "model_prompts": list(emb.prompts),
        "dim": args.dim,
        "web_dim": args.web_dim,
        "n": int(len(s)),
        "n_total": int(len(index)),
        "strategy": args.strategy,
        "seed": args.seed,
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "thumb": THUMB,
        "sheet_cols": SHEET_COLS,
        "sheets": sheets,
        "local_crops": bool(args.copy_crops),
        "k": args.k,
        "clusters": k_clusters,
        "prompt_template": L.PROMPT_TEMPLATE,
        "classes": [{"code": c, "name": (t := L.info(c))["name"], "group": group_codes.index(t["t1"]),
                     "t2": t["t2"], "t2_name": t["t2_name"], "t2b": t["t2b"], "t2b_name": t["t2b_name"],
                     "prompt": t["prompt"], "sure": t["sure"], "n": int((y == i).sum()),
                     "n_total": int(total.get(c, 0))} for i, c in enumerate(codes)],
        "groups": [{"code": g, "name": L.T1_GROUPS[g][0]} for g in group_codes],
        "projections": list(proj),
        "metrics": {
            "linear_probe_acc": float((lp_idx[:, 0] == y).mean()),
            "linear_probe_group_acc": float((proba_g.argmax(1) == yg).mean()),
            "knn_acc": float((knn_pred == y).mean()),
            "zero_shot_acc": float((zs_idx[:, 0] == y).mean()),
            "zero_shot_group_acc": float((zs_g.argmax(1) == yg).mean()),
            "coralnet_agree": float((s.cn_pred == s.label).mean()),
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT / "points.json").write_text(json.dumps(points, separators=(",", ":")))
    log("metrics: " + json.dumps({k: round(v, 3) for k, v in manifest["metrics"].items()}))
    log(f"wrote {OUT} ({sum(f.stat().st_size for f in OUT.rglob('*')) / 1e6:.1f} MB)")

    if args.export:
        dest = Path(args.export)
        log(f"exporting parquet + zarr to {dest} ...")
        meta = export.metadata_frame(s, {c: L.info(c) for c in codes}, gcs.BUCKET, cluster, proj)
        export.write_dataset(dest, meta, full, {
            "model": emb.name, "model_is_stub": args.embedder == "stub",
            "dataset": manifest["dataset"], "bucket": gcs.BUCKET, "built": manifest["built"],
            "strategy": args.strategy, "seed": args.seed})
        log(f"wrote {dest} ({sum(f.stat().st_size for f in dest.rglob('*')) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
