"""Export a build as a cloud-native dataset that can be read in pieces over HTTP.

    <dir>/metadata.parquet   one row per patch: labels, taxonomy, survey fields, layout coords
    <dir>/embeddings.zarr    float16 array (n_patches, dim), row i = metadata row i

Upload the directory to any host that serves HTTP range requests (a GCS/S3 bucket, a
Hugging Face dataset repo) and read it without downloading everything:

    meta = pd.read_parquet(f"{base}/metadata.parquet")
    emb = zarr.open_array(f"{base}/embeddings.zarr", mode="r")   # needs fsspec + aiohttp
    v = emb[1234]                                                 # fetches one chunk

Embeddings are L2-normalised at full dimension. For a Matryoshka prefix, slice the first
`d` columns and re-normalise.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_ROWS = 1024  # 1024 x 768 float16 = 1.5 MB per request


def metadata_frame(s: pd.DataFrame, info: dict, bucket: str, cluster: np.ndarray,
                   proj: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per patch. `info` maps class code -> labels.info(code)."""
    def num(c):
        return pd.to_numeric(s[c], errors="coerce") if c in s else pd.Series(np.nan, index=s.index)

    def text(c):
        return s[c].astype("string") if c in s else pd.Series(pd.NA, index=s.index, dtype="string")

    df = pd.DataFrame({
        "row": np.arange(len(s), dtype=np.int32),
        "gcs_uri": [f"gs://{bucket}/{o}" for o in s.object],
        "object": s.object.astype("string"),
        "file": s.file.astype("string"),
        "split": s.split.astype("string"),
        "label": s.label.astype("string"),
        "label_name": [info[c]["name"] for c in s.label],
        "t1": [info[c]["t1"] for c in s.label],
        "t2": [info[c]["t2"] for c in s.label],
        "t2b": [info[c]["t2b"] for c in s.label],
        "site": s.site.astype("string"),
        "island_code": s.island_code.astype("string"),
        "year": pd.array(s.year, dtype="Int16"),
        "image_name": s.image_name.astype("string"),
        "point": s.point.astype(np.int32),
        "source": text("source"),
        "region": text("Region"),
        "island": text("Island"),
        "date": text("Date"),
        "lat": num("Latitude"),
        "lon": num("Longitude"),
        "depth_m": num("Depth"),
        "photo_row": pd.array(num("Row"), dtype="Int32"),
        "photo_col": pd.array(num("Column"), dtype="Int32"),
        "coralnet_pred": text("cn_pred"),
        "coralnet_conf": num("cn_conf"),
        "cluster": cluster.astype(np.int16),
    })
    for name, p in proj.items():
        if name == "geo" or p.shape[1] != 2:
            continue
        df[f"{name}_x"], df[f"{name}_y"] = p[:, 0].astype(np.float32), p[:, 1].astype(np.float32)
    return df


def write_dataset(out: Path, meta: pd.DataFrame, emb: np.ndarray, attrs: dict) -> None:
    import zarr

    assert len(meta) == len(emb), "metadata and embeddings must line up row for row"
    out.mkdir(parents=True, exist_ok=True)
    attrs = {**attrs, "n": int(len(emb)), "dim": int(emb.shape[1]), "normalized": True,
             "rows": "row i matches metadata.parquet row i"}

    table = pa.Table.from_pandas(meta, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"tide_pool": json.dumps(attrs).encode()})
    pq.write_table(table, out / "metadata.parquet", compression="zstd", row_group_size=8192)

    path = out / "embeddings.zarr"
    if path.exists():
        shutil.rmtree(path)
    # uncompressed: embeddings barely compress, and plain chunks are simplest for any reader
    arr = zarr.create_array(store=str(path), shape=emb.shape, dtype="float16",
                            chunks=(min(CHUNK_ROWS, len(emb)), emb.shape[1]),
                            compressors=None, attributes=attrs)
    arr[:] = emb.astype(np.float16)
