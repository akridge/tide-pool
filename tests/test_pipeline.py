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


def test_every_class_has_a_known_group():
    assert all(g in labels.T1_GROUPS for g, _, _ in labels.T3.values())
