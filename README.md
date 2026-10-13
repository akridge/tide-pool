# Tide Pool

An interactive explorer for **[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)** image embeddings of the
**[NOAA Pacific benthic cover, Tier 3](https://huggingface.co/datasets/NMFS-OSI/noaa-pacific-benthic-cover-t3)** patches.
It shows more than the embedding cloud. You can also see the image patches themselves, the full survey photo each patch was cut from, model diagnostics and the survey metadata.

```
pipeline/build.py   sample → download → join metadata → embed → analyse → web/data/
pipeline/export.py  optional Parquet + Zarr export of the same build (--export DIR)
pipeline/full.py    embeds and analyses ALL 378,680 patches for the WebGL viewer at /full/
server/app.py       serves web/ and embeds text / image queries with the same model
web/                dependency-free canvas app (no build step)
docker-compose.yml  build / app / pages services (docker-compose.gpu.yml for NVIDIA)
```

## Quick start

```bash
pip install -r requirements.txt
huggingface-cli login                       # if the model requires accepting its terms

python pipeline/build.py --n 6000           # ~6k patches, balanced across 67 classes
python server/app.py                        # → http://localhost:8000
```

On a GPU, 6,000 patches embed in a minute or two. On CPU it takes longer, roughly 10–20 minutes. Every stage caches its output under
`data/cache/`, so if you raise `--n` later, only the new patches are downloaded and embedded.

To check the UI without downloading the model, run `python pipeline/build.py --n 600 --embedder stub`. The stub uses
colour and texture features, not semantic embeddings, and the app shows a banner while it's in use.

## All patches (`/full/`)

`pipeline/build.py` works on a sample, a few thousand to a few tens of thousands of patches, and ships everything to
the browser. To embed **every** patch, use `pipeline/full.py` and open the WebGL viewer at
http://localhost:8000/full/:

```bash
python pipeline/full.py                      # all 378,680 patches with EmbeddingGemma 2 (GPU strongly recommended)
python server/app.py                         # restart it after each run so it loads the new vectors
python pipeline/full.py --limit 20000 --embedder stub   # quick try-out on a random subset
```

With Docker (Windows `cmd`: use `set` instead of `export` and `;` instead of `:` in `COMPOSE_FILE`):

```bash
export COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
docker compose run --rm full
docker compose up -d app                     # → http://localhost:8000/full/
```

How it scales:

- **Streaming, resumable embedding.** Crops are downloaded in shards of 4,096 (the next shard downloads while the GPU
  embeds the current one), embedded and dropped, so the 32 GB of crops never lands on disk. Each finished shard is
  saved to `data/full/emb_<model>/`. Stop the run at any time and start it again: finished shards are skipped.
- **Analysis at full size.** Exact k-NN runs on the GPU through torch, and falls back to the CPU, much more slowly.
  It also computes neighbour agreement, the k-NN vote, zero-shot top 3, mini-batch k-means (64 clusters), PCA, and UMAP
  on a 50-d PCA. The result is cached in `data/full/analysis_<model>.npz`; `--reanalyse` recomputes it and
  `--skip-umap` skips the slowest step.
- **Viewer.** The browser loads compact binary columns, about 60 bytes per patch (about 25 MB for every patch), and
  draws all the points with WebGL. A patch's details and neighbours are fetched only when you open it. Patch images
  load from NOAA's bucket on hover, in the detail panel, and as tiles in *Patches* mode once fewer than 450 are on screen.
- **Search over every vector.** Free-text search, image search and *Find similar* (`S`) run in the server against
  `data/full/search.npy` (256-d, about 190 MB) and colour all the points by similarity. Typing a class code
  (e.g. `POMA`) filters to that class, with or without the server.

Disk use for every patch: about 600 MB of embedding shards, 190 MB for the search matrix, and roughly 200 MB of viewer
data, most of it per-patch detail files. The full run has only been tested on a 20,000-patch subset with the stub
embedder; GPU speed for EmbeddingGemma 2 depends on your card. The viewer has no linear probe, t-SNE, 3-D view or
classes/confusions tabs; those stay in the sample explorer at `/`.

## Docker

```bash
export HF_TOKEN=hf_...                                   # if the model requires accepting its terms
docker compose run --rm build --n 6000                   # writes ./web/data, caches in ./data
docker compose up -d app                                 # → http://localhost:8000 (with search)
docker compose --profile pages up pages                  # → http://localhost:8080 (static, as on GitHub Pages)
```

`build` takes the same flags as `pipeline/build.py` (for a quick test without the model: `--n 600 --embedder stub`),
and `full` takes the flags of `pipeline/full.py` (see [All patches](#all-patches-full)).
The image uses CPU torch. On an NVIDIA GPU (with the NVIDIA Container Toolkit installed), add the override file to every
command: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm build --n 6000`. Downloaded model
weights are kept in the `hf-cache` volume, so they aren't downloaded again.

## Hosting on GitHub Pages

A build is self-contained: everything the app needs is in `web/`, so `web/` can be served by any static host.
Patch crops and full survey photos still load from NOAA's public bucket. Free-text and image search need the server, so
on a static host they are off and search falls back to class names.

`web/data/` is git-ignored, so publish it explicitly, for example from a `gh-pages` branch (needs git 2.42 or newer):

```bash
python pipeline/build.py --n 6000                        # or the docker compose build above
git worktree add --orphan -b gh-pages ../tide-pool-pages
cp -r web/. ../tide-pool-pages/ && touch ../tide-pool-pages/.nojekyll
cd ../tide-pool-pages && git add -A && git commit -m "Publish bundle" && git push -u origin gh-pages
```

Then set Settings → Pages → *Deploy from a branch* → `gh-pages` / root.

Only the sampled patches are embedded and shipped, not all 378,680. A bundle costs about 3 KB per patch (1 KB of
that is the 256-d embedding), so 6,000 patches is about 18 MB and 30,000 is about 90 MB. GitHub rejects files over
100 MB, which `embeddings.f32` reaches at about 100,000 patches with `--web-dim 256`, or about 200,000 with
`--web-dim 128`. Pages sites are limited to 1 GB. Every visitor also downloads the whole bundle, so a few tens of
thousands of patches is a practical ceiling.

## Cloud-native export (Parquet + Zarr)

`--export DIR` also writes the build as a dataset that can be read in pieces over HTTP:

```
DIR/metadata.parquet   one row per patch: gcs_uri, split, label, T1/T2/T2b, site, island, date, lat/lon,
                       point position in the photo, CoralNet suggestion, k-means cluster, UMAP/PCA/t-SNE coords
DIR/embeddings.zarr    float16 (n_patches, 768), L2-normalised, 1,024-row chunks; row i = metadata row i
```

```bash
python pipeline/build.py --n 6000 --export data/export
docker compose run --rm build --n 6000 --export data/export      # ./data is mounted, so it lands on the host
```

Upload `DIR` to a bucket or a Hugging Face dataset repo, then read only what you need:

```python
import pandas as pd, zarr                     # remote reads also need fsspec and aiohttp
base = "https://huggingface.co/datasets/<you>/<repo>/resolve/main"
meta = pd.read_parquet(f"{base}/metadata.parquet")
emb = zarr.open_array(f"{base}/embeddings.zarr", mode="r")
v = emb[1234]                                 # downloads one 1.5 MB chunk, not the whole array
```

Each Zarr chunk is its own file, so any static host works, with no range-request support needed. For a shorter
Matryoshka vector, take the first `d` columns and re-normalise. With `--n 378680` the export covers the whole dataset
(about 580 MB of embeddings). The browser bundle in `web/data` should stay much smaller (see above).

## Data source

The pipeline reads the public bucket behind the HF dataset directly, with no credentials:

```
gs://nmfs_odp_pifsc/PIFSC/ESD/ARP/pifsc-ai-data-repository/coralnet_mirror/
  pacific_t3_splits/{train,val,test}/<T3 code>/<code>_<site>_<year>_A_<photo>_<point>.png   378,680 crops, 224×224
  pacific_t1/<T1 code>/…                       same files grouped by the 8 broad classes (gives the T3→T1 map)
  ESD_REA_{HAWAII,MARIANA,SAMOA_PRIA}_v2/
    *_annotation(s).csv                         per-point metadata
    images/<site>_<year>_A_<photo>.JPG          full-resolution quadrat photos
```

Each crop is joined to its annotation row by photo name and point order. The `_<point>` suffix is the point's 1-based
position within its photo in the CSV, and the crop is centred exactly on that row's `Row`/`Column`. About 99.8% of
crops join, which adds lat/lon, region, island, site, date and CoralNet's top-5 machine suggestions. The source CSVs
have a `Depth` column, but it is empty in all three, so depth shows as blank in the app and the export.

## What the app shows

**Views:** UMAP 2D, UMAP 3D (orbit), t-SNE, PCA, and a **Map** that plots the survey coordinates (lon wrapped across the antimeridian).

**Rendering:** dots, or the **image patches themselves** drawn from sprite sheets. Patch borders are coloured by the
current encoding.

**Colour by:**

| | |
|---|---|
| Labels | T1 group, T2 growth form, T2b genus, T3 class (hue = T1 group, lightness = finer level) |
| Model behaviour | search similarity, neighbour-label agreement, linear-probe / k-NN / zero-shot correctness and predicted group, probe confidence, atypicality, similarity to class centroid, k-means cluster |
| Survey metadata | split, region, island, year, depth, whether CoralNet's suggestion agrees, CoralNet confidence |
| Pixels | true mean colour, dominant hue, brightness, contrast, saturation, colourfulness, sharpness, texture entropy |

**Interactions:**

- **Hover** for a thumbnail.
- **Click** for the detail panel: the 224px patch, survey fields, linear-probe top-5, zero-shot top-5 against class text descriptions, CoralNet suggestions, the 12 nearest neighbours (red outline = different label), and pixel statistics.
- **Show in photo** opens the full quadrat photo, zoomed to the point, with every sampled patch from that photo outlined.
- **Shift-drag** to lasso a set of patches into a gallery with its class composition. You can export it to CSV.
- **Search:** with the server running, search by free text (EmbeddingGemma's `SearchQuery` prompt) or by uploading an image. Without the server, a class code or name is searched with its build-time text prototype. **Find similar** (or the `s` key) ranks every patch by cosine similarity to the selected one.

**Tabs:**

- **Classes:** sortable per-class table of neighbour agreement, probe, k-NN, zero-shot and CoralNet accuracy.
- **Confusions:** T1 matrix plus the most frequent T3 confusions for each predictor, all clickable.
- **Clusters:** k-means cards showing purity and their most typical patches.
- **About:** headline metrics.

## Model notes

- Loaded through `sentence-transformers>=6.1` with `config_kwargs={"audio_config": None}`, so only the text and vision
  encoders are loaded. If the loader rejects that option, the pipeline falls back to the full model.
- Uses bf16 on CUDA and fp32 elsewhere, because the model card warns that fp16 overflows.
- Images are encoded as `{"image": PIL.Image}` with no prompt. Class descriptions use the `Document` prompt and search
  queries use `SearchQuery`, when the model exposes those prompt names.
- Matryoshka: `--dim` (768/512/256/128) sets the dimension used for analysis. `--web-dim` (default 256) sets what's shipped to the
  browser for client-side similarity. Vectors are truncated and then re-normalised.
- The linear probe is out-of-fold logistic regression with `StratifiedGroupKFold` grouped by source photo, so
  patches from one photo never sit on both sides of a fold.

## Build options

```
--n 6000                  patches to sample          --strategy balanced|proportional
--splits train,val,test   which splits to draw from  --seed 7
--dim 768 --web-dim 256   Matryoshka dimensions      --k 12  neighbours per patch
--clusters 0              k-means k (0 = auto)       --skip-tsne
--copy-crops              bundle 224px crops locally (otherwise loaded from the public bucket)
--export DIR              also write metadata.parquet + embeddings.zarr (see above)
--labels-csv names.csv    override class names (code,name[,group])
--embedder gemma|stub     --model google/embeddinggemma-2   --device cuda|mps|cpu
```

Class names and the hierarchy come from the PIFSC taxonomy tree in `pipeline/taxonomy_tree.json`:
T1 broad group → T2 growth form / type (encrusting, branching, massive, upright macroalga, …) → T2b genus / subgroup
(Porites sp, Montipora sp, …) → T3 class. The app can colour by T2 and T2b as well as T1 and T3. The zero-shot prompts use
the plain-English phrases in `PROMPT_NAMES` in `pipeline/labels.py`, which follow the official names. A code
missing from the tree is shown with "?" in the app, and a test fails if any dataset code is missing. To rename classes for
one build, pass `--labels-csv`.

`web/` is static, so `web/` plus a built `web/data/` can be hosted anywhere, GitHub Pages included. Without the server, free-text and
image search are disabled.

## Tests

```bash
pip install pytest
python -m pytest tests
```
