"""Unit tests for hybrid_pipeline.py — the quality gate and the JSON/
narrative parser, which are the two purely deterministic pieces of Task 4
that don't require a live Ollama server to test."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuclei_pipeline.hybrid_pipeline import quality_gate, parse_hybrid_response


def make_df(n=10, area=100.0):
    return pd.DataFrame({"area": [area] * n})


def test_quality_gate_passes_normal_case():
    df = make_df(n=20, area=100.0)
    passed, reason = quality_gate(df)
    assert passed
    assert reason == "ok"


def test_quality_gate_rejects_zero_objects():
    df = make_df(n=0)
    passed, reason = quality_gate(df)
    assert not passed
    assert "no objects" in reason


def test_quality_gate_rejects_too_many_objects():
    df = make_df(n=200, area=100.0)
    passed, reason = quality_gate(df)
    assert not passed
    assert "too many" in reason


def test_quality_gate_rejects_tiny_mean_area():
    df = make_df(n=10, area=5.0)
    passed, reason = quality_gate(df)
    assert not passed
    assert "too small" in reason


def test_quality_gate_boundary_is_inclusive_at_min_objects():
    df = make_df(n=1, area=100.0)
    passed, reason = quality_gate(df, min_objects=1)
    assert passed


def test_parse_hybrid_response_well_formed():
    text = (
        'JSON:\n{"image_id": "x", "n_objects": 5, "density_class": "sparse", '
        '"shape_regularity": "regular", "size_uniformity": "uniform", '
        '"quality_flag": "ok", "confidence": "high"}\n\n'
        "NARRATIVE:\nFive round objects were detected, evenly sized."
    )
    record, narrative = parse_hybrid_response(text)
    assert record["n_objects"] == 5
    assert record["density_class"] == "sparse"
    assert narrative.startswith("Five round objects")


def test_parse_hybrid_response_strips_code_fences():
    text = 'JSON:\n```json\n{"n_objects": 3}\n```\n\nNARRATIVE:\nThree objects found.'
    record, narrative = parse_hybrid_response(text)
    assert record["n_objects"] == 3
    assert narrative == "Three objects found."


def test_parse_hybrid_response_malformed_json_does_not_raise():
    text = "JSON:\n{not valid json at all\n\nNARRATIVE:\nSomething went wrong."
    record, narrative = parse_hybrid_response(text)
    assert record == {"error": "could not parse JSON"}
    assert "Something went wrong" in narrative


def test_parse_hybrid_response_missing_markers_still_returns_tuple():
    text = "just some free text with no markers"
    record, narrative = parse_hybrid_response(text)
    assert isinstance(record, dict)
    assert isinstance(narrative, str)
