import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import build  # noqa: E402
import labels  # noqa: E402


def test_parse_name():
    r = build.parse_name("x/pacific_t3_splits/train/ACBR/ACBR_KUR-4255_2024_A_17_7.png")
    assert r["split"] == "train" and r["label"] == "ACBR"
    assert r["image_name"] == "KUR-4255_2024_A_17.JPG"
    assert r["point"] == 7 and r["site"] == "KUR-4255" and r["island_code"] == "KUR" and r["year"] == 2024


def test_parse_name_rejects_non_png_and_mismatched_prefix():
    assert build.parse_name("x/pacific_t3_splits/") is None
    assert build.parse_name("x/train/ACBR/POCS_KUR-1_2024_A_01_1.png") is None


def test_balanced_sample_respects_budget_and_small_classes():
    df = pd.DataFrame({"object": [f"o{i}" for i in range(130)],
                       "label": ["A"] * 100 + ["B"] * 25 + ["C"] * 5,
                       "split": ["train"] * 130})
    s = build.sample(df, 60, "balanced", ["train"], seed=0)
    counts = s.label.value_counts()
    # C is exhausted, its unused share flows to B (also exhausted) and then to A
    assert len(s) == 60 and counts.to_dict() == {"A": 30, "B": 25, "C": 5}


def test_matryoshka_truncates_and_renormalises():
    x = np.random.default_rng(0).standard_normal((4, 768)).astype(np.float32)
    y = build.matryoshka(x, 256)
    assert y.shape == (4, 256)
    assert np.allclose(np.linalg.norm(y, axis=1), 1, atol=1e-5)


def test_knn_excludes_self_and_sorts():
    x = build.matryoshka(np.random.default_rng(1).standard_normal((50, 16)), 16)
    idx, sim = build.knn(x, 5)
    assert not (idx == np.arange(50)[:, None]).any()
    assert (np.diff(sim, axis=1) <= 1e-6).all()


# the 67 T3 codes in pacific_t3_splits/ (each has a hand-written zero-shot phrase)
DATASET_CODES = sorted(labels.PROMPT_NAMES)


def test_every_dataset_class_is_in_the_taxonomy_tree():
    assert len(DATASET_CODES) == 67
    missing = [c for c in DATASET_CODES if c not in labels.TAXONOMY]
    assert not missing, f"add these codes to pipeline/taxonomy_tree.json: {missing}"
    assert all(labels.info(c)["sure"] for c in DATASET_CODES)


def test_every_class_has_a_known_group():
    assert all(labels.group_of(c) in labels.T1_GROUPS for c in DATASET_CODES)


def test_tree_gives_official_names_and_levels():
    poma = labels.info("POMA")
    assert (poma["t1"], poma["t2"], poma["t2b"], poma["name"]) == ("CORAL", "MASS", "POSP", "Porites massive")
    # DICO and DICT are easy to swap: DICO is the brown Dictyopteris/Dictyota, DICT the green Dictyosphaeria
    assert labels.name_of("DICO") == "Dictyopteris/Dictyota" and labels.info("DICO")["t2b"] == "BRMA"
    assert labels.name_of("DICT") == "Dictyosphaeria sp"
    assert "Dictyosphaeria" in labels.prompt_of("DICT") and "Dictyota" in labels.prompt_of("DICO")


def test_unknown_code_falls_back_and_is_unsure():
    i = labels.info("NOPE")
    assert i["name"] == "NOPE" and i["t1"] == "UNK" and not i["sure"]


def test_labels_csv_override(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "_overrides", {})
    f = tmp_path / "names.csv"
    f.write_text("code,name\nSP,demosponge\n")
    labels.load_overrides(str(f))
    assert labels.name_of("SP") == "demosponge" and labels.prompt_of("SP") == "demosponge"
    assert labels.group_of("SP") == "I"
