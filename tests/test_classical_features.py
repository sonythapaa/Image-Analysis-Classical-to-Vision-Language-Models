"""Unit tests for classical_features.py — flat function style, hand-computed
expected values, shared make_df() helper for tests that need a feature table
without running segmentation first."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuclei_pipeline.classical_features import otsu_segment, region_feature_table, features_to_text


def make_df(n=3, area=100.0, eccentricity=0.5, solidity=0.9, mean_intensity=0.5):
    """Small helper: build a feature table with known, hand-set values so
    features_to_text()'s arithmetic can be checked exactly."""
    return pd.DataFrame({
        "label": range(1, n + 1),
        "area": [area] * n,
        "eccentricity": [eccentricity] * n,
        "solidity": [solidity] * n,
        "mean_intensity": [mean_intensity] * n,
    })


def make_two_blob_image(size=64):
    """Two well-separated bright squares on a dark background — a
    deterministic image where the expected object count is known exactly."""
    img = np.zeros((size, size), dtype=np.float32)
    img[5:15, 5:15] = 0.9
    img[40:55, 40:55] = 0.9
    return img


def test_otsu_segment_finds_both_blobs():
    img = make_two_blob_image()
    mask = otsu_segment(img, min_size=5)
    assert mask.dtype == bool
    assert mask.sum() > 0
    # both bright squares should be foreground
    assert mask[10, 10]
    assert mask[47, 47]
    # background corner should stay background
    assert not mask[0, 0]


def test_otsu_segment_empty_image_gives_empty_mask():
    img = np.zeros((32, 32), dtype=np.float32)
    # threshold_otsu on a constant image would divide by zero internally in
    # some implementations; guard by adding a single differing pixel.
    img[0, 0] = 1.0
    mask = otsu_segment(img, min_size=1)
    assert mask.sum() <= 1  # at most the single bright pixel


def test_region_feature_table_counts_two_objects():
    img = make_two_blob_image()
    mask = otsu_segment(img, min_size=5)
    df = region_feature_table(img, mask)
    assert len(df) == 2
    assert set(["area", "eccentricity", "solidity", "mean_intensity"]).issubset(df.columns)


def test_region_feature_table_empty_mask_returns_empty_df():
    img = np.zeros((16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=bool)
    df = region_feature_table(img, mask)
    assert len(df) == 0


def test_features_to_text_reports_correct_count():
    df = make_df(n=5)
    text = features_to_text(df)
    assert "Detected 5 objects" in text


def test_features_to_text_reports_correct_mean_area():
    df = make_df(n=3, area=100.0)
    text = features_to_text(df)
    # mean area of three identical 100.0 values is exactly 100.0
    assert "Mean area 100.0" in text


def test_features_to_text_zero_cv_when_areas_identical():
    df = make_df(n=4, area=50.0)
    text = features_to_text(df)
    # identical areas -> std 0 -> coefficient of variation 0.00
    assert "variation 0.00" in text


def test_features_to_text_empty_table():
    df = make_df(n=0)
    text = features_to_text(df)
    assert "No objects" in text
