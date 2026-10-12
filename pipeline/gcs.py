"""Anonymous access to the public NOAA PIFSC CoralNet mirror on Google Cloud Storage."""
from __future__ import annotations

import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import requests

BUCKET = "nmfs_odp_pifsc"
MIRROR = "PIFSC/ESD/ARP/pifsc-ai-data-repository/coralnet_mirror"
T3_PREFIX = f"{MIRROR}/pacific_t3_splits"
T1_PREFIX = f"{MIRROR}/pacific_t1_splits"
# Survey sources whose annotation CSVs carry per-point metadata (lat/lon, depth, island, ...)
SOURCES = {
    "ESD_REA_HAWAII_v2": "ESD_REA_HAWAII_v2_annotations.csv",
    "ESD_REA_MARIANA_v2": "ESD_REA_MARIANA_v2_annotations.csv",
    "ESD_REA_SAMOA_PRIA_v2": "ESD_REA_SAMOA_PRIA_v2_annotation.csv",
}

API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
PUBLIC = "https://storage.googleapis.com/{bucket}/{name}"

_session = requests.Session()


def public_url(name: str, bucket: str = BUCKET) -> str:
    return PUBLIC.format(bucket=bucket, name=urllib.parse.quote(name))


def _get(url: str, **kw) -> requests.Response:
    for attempt in range(5):
        try:
            r = _session.get(url, timeout=60, **kw)
            if r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
                return r
        except requests.ConnectionError:
            pass
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r


def list_objects(prefix: str, bucket: str = BUCKET) -> Iterator[tuple[str, int]]:
    """Yield (object name, size) for every object under prefix."""
    params = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": 5000}
    while True:
        data = _get(API.format(bucket=bucket), params=params).json()
        for it in data.get("items", []):
            yield it["name"], int(it["size"])
        tok = data.get("nextPageToken")
        if not tok:
            return
        params["pageToken"] = tok


def download(name: str, dest: Path, bucket: str = BUCKET) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _get(public_url(name, bucket), stream=True) as r:
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def download_many(pairs: list[tuple[str, Path]], workers: int = 16) -> list[Path | None]:
    def one(p):
        try:
            return download(*p)
        except Exception as e:  # keep going; caller drops failures
            print(f"  ! failed {p[0]}: {e}")
            return None

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, pairs))
