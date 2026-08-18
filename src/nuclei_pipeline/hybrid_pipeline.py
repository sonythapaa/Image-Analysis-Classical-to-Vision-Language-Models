"""the full hybrid pipeline.

    raw image -> U-Net mask -> regionprops feature table
              -> LLM JSON record + narrative -> aggregated CSV

Design principles taken straight from Lecture 5 ("Hybrid Pipelines and
Medical AI in Practice"):
  1. Push work into the auditable (deterministic) component; the LLM only
     narrates numbers it is given, it never measures anything itself.
  2. The narrative may not invent facts — every number in it should trace
     back to the structured JSON record.
  3. A cheap, deterministic quality gate runs BEFORE the (slow, stochastic)
     LLM call, so a bad image never reaches the narrator.
  4. An audit column (`n_objects_measured`, from regionprops) rides beside
     the LLM's own `n_objects` field so a human reviewer can catch
     hallucination by comparing the two.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from ollama import chat
except ImportError:  # pragma: no cover
    chat = None

from .classical_features import otsu_segment, region_feature_table, features_to_text, mask_dice, mask_iou
from .unet import UNet

MODEL = "llama3.2"

HYBRID_PROMPT_TEMPLATE = """You are summarising a biomedical image for a research report.

You will be given a textual summary of the image. Based ONLY on that summary:

1. Produce a JSON record with these fields:
   - image_id (string)
   - n_objects (integer)
   - mean_area (number, in pixels — the average area of detected objects)
   - density_class ("sparse", "moderate", "dense")
   - shape_regularity ("highly irregular", "irregular", "regular", "highly regular")
   - size_uniformity ("uniform", "mixed", "highly variable")
   - quality_flag ("ok", "review_recommended", "fail")
   - confidence ("high", "medium", "low")

2. Then a one-paragraph narrative (3-4 sentences), suitable for a research report.

Format your response EXACTLY as:

JSON:
{{...}}

NARRATIVE:
<paragraph>

Rules:
- Do NOT invent details that are not in the summary.
- Do NOT diagnose any medical condition.
- If anything is unclear, lower the confidence field.

Image ID: {image_id}
Summary: {summary}"""


# Stage 1: segmentation (U-Net, falls back to Otsu if no trained weights)

def load_unet(weights_path: str | Path, device=None) -> UNet:
    device = device or torch.device("cpu")
    model = UNet(base=16).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def unet_segment(model: UNet, gray: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Run the trained U-Net on a single grayscale [0,1] image -> boolean mask."""
    with torch.no_grad():
        x = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0]
        return (probs > threshold).numpy()


# Stage 2/3: measurement + text summary (reuses the classical_features)

def measure(gray: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    return region_feature_table(gray, mask)


def summarise(df: pd.DataFrame, image_id: str) -> str:
    return f"Image {image_id}. " + features_to_text(df)


# Stage 3.5: quality gate — cheap, deterministic, runs BEFORE the LLM

def quality_gate(df: pd.DataFrame, min_objects: int = 1, max_objects: int = 150,
                  min_mean_area: float = 15.0) -> tuple[bool, str]:
    """Reject an image before spending an LLM call on it. Pure, deterministic."""
    n = len(df)
    if n < min_objects:
        return False, "no objects detected"
    if n > max_objects:
        return False, "too many objects (likely noise / over-segmentation)"
    if df["area"].mean() < min_mean_area:
        return False, "mean object area too small (mask is mostly speckle)"
    return True, "ok"


# Stage 4: LLM — JSON record + narrative in one call

def query_llm(image_id: str, summary: str, model_name: str = MODEL, temperature: float = 0.0) -> str:
    if chat is None:
        raise RuntimeError(
            "The `ollama` package / server isn't available in this environment. "
            "Run this where Ollama is installed, as in Lab 5."
        )
    prompt = HYBRID_PROMPT_TEMPLATE.format(image_id=image_id, summary=summary)
    response = chat(model=model_name, messages=[{"role": "user", "content": prompt}],
                     options={"temperature": temperature})
    return response["message"]["content"]


def parse_hybrid_response(text: str) -> tuple[dict, str]:
    """Split the combined 'JSON:/NARRATIVE:' reply into a dict + prose,
    failing gracefully (never raising) on malformed model output."""
    text = text.strip()
    after_json = text.split("JSON:", 1)[1] if "JSON:" in text else text
    if "NARRATIVE:" in after_json:
        json_part, narrative_part = after_json.split("NARRATIVE:", 1)
    else:
        json_part, narrative_part = after_json, ""

    json_part = json_part.strip()
    if json_part.startswith("```"):
        json_part = json_part.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        record = json.loads(json_part)
    except json.JSONDecodeError:
        record = {"error": "could not parse JSON"}
    return record, narrative_part.strip()



# Full pipeline: wraps every stage, with the quality gate short-circuiting

def full_pipeline(gray: np.ndarray, image_id: str, unet_model: UNet,
                   model_name: str = MODEL, use_gate: bool = True) -> dict:
    mask = unet_segment(unet_model, gray)
    df = measure(gray, mask)

    if use_gate:
        passed, reason = quality_gate(df)
        if not passed:
            return {
                "image_id": image_id,
                "n_objects": len(df),
                "n_objects_measured": len(df),
                "mean_area": round(df["area"].mean(), 1) if len(df) else 0.0,
                "mean_area_measured": round(df["area"].mean(), 1) if len(df) else 0.0,
                "density_class": "uncertain",
                "shape_regularity": "uncertain",
                "size_uniformity": "uncertain",
                "quality_flag": "fail",
                "confidence": "high",
                "gate_reason": reason,
                "narrative": "Image rejected at quality gate; LLM was not called.",
            }

    summary = summarise(df, image_id)
    raw = query_llm(image_id, summary, model_name=model_name)
    record, narrative = parse_hybrid_response(raw)
    record["image_id"] = image_id
    record["n_objects_measured"] = len(df)
    record["mean_area_measured"] = round(df["area"].mean(), 1) if len(df) else 0.0
    record["gate_reason"] = "ok"
    record["narrative"] = narrative
    return record


def run_batch(image_dir: str | Path, unet_model: UNet, model_name: str = MODEL) -> pd.DataFrame:
    """run the full pipeline on every unseen test image
    and aggregate the per-image JSON records into one DataFrame."""
    from .data_prep import load_and_preprocess

    image_dir = Path(image_dir)
    records = []
    for p in sorted(image_dir.glob("*.png")):
        gray = load_and_preprocess(p)
        rec = full_pipeline(gray, p.stem, unet_model, model_name=model_name)
        records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Task 3 discussion material: does the U-Net's near-perfect pixel Dice
# actually translate into correct object COUNTS? Semantic segmentation
# merges touching objects into one blob, so regionprops on the predicted
# mask can undercount relative to the ground-truth instance count in
# metadata.csv even when Dice/IoU look excellent.
# ---------------------------------------------------------------------------

def instance_merging_analysis(unet_model: UNet, metadata: pd.DataFrame, image_dir: str | Path,
                                split_name: str = "test") -> pd.DataFrame:
    """For every image in `image_dir`, compare the ground-truth object count
    (metadata.csv, from the dataset's own instance labels) against the
    object count obtained by running regionprops on the U-Net's predicted
    mask. Returns a DataFrame sorted by the size of the undercount."""
    from .data_prep import load_and_preprocess

    image_dir = Path(image_dir)
    rows = []
    for p in sorted(image_dir.glob("*.png")):
        gray = load_and_preprocess(p)
        mask = unet_segment(unet_model, gray)
        df = measure(gray, mask)
        gt_row = metadata[metadata.image_id == p.stem].iloc[0]
        n_gt = int(gt_row["n_objects"])
        n_measured = len(df)
        rows.append({
            "image_id": p.stem,
            "density": gt_row["density"],
            "n_objects_ground_truth": n_gt,
            "n_objects_measured": n_measured,
            "undercount": n_gt - n_measured,
            "undercount_pct": round((n_gt - n_measured) / n_gt * 100, 1) if n_gt > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("undercount_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Robustness extension: trace a corrupted image through every stage and
# find the earliest point at which the corruption becomes detectable.
# ---------------------------------------------------------------------------

def robustness_trace(unet_model: UNet, corrupted_dir: str | Path, clean_dir: str | Path) -> pd.DataFrame:
    """For every corrupted image (named `<base_id>_<variant>.png`, e.g.
    `test_000_blur.png`), run the deterministic pipeline stages on both the
    corrupted version and its clean counterpart and compare. Returns one
    row per corrupted image with enough detail to say, in the report,
    exactly which stage first shows a detectable difference."""
    from .data_prep import load_and_preprocess

    corrupted_dir, clean_dir = Path(corrupted_dir), Path(clean_dir)
    rows = []
    for cf in sorted(corrupted_dir.glob("*.png")):
        base_id, variant = cf.stem.rsplit("_", 1)
        clean_path = clean_dir / f"{base_id}.png"
        if not clean_path.exists():
            continue

        gray_clean = load_and_preprocess(clean_path)
        gray_corrupt = load_and_preprocess(cf)
        mask_clean = unet_segment(unet_model, gray_clean)
        mask_corrupt = unet_segment(unet_model, gray_corrupt)
        df_clean = measure(gray_clean, mask_clean)
        df_corrupt = measure(gray_corrupt, mask_corrupt)
        passed_clean, reason_clean = quality_gate(df_clean)
        passed_corrupt, reason_corrupt = quality_gate(df_corrupt)

        rows.append({
            "base_image": base_id,
            "variant": variant,
            "n_objects_clean": len(df_clean),
            "n_objects_corrupted": len(df_corrupt),
            "mean_area_clean": round(df_clean["area"].mean(), 1) if len(df_clean) else 0.0,
            "mean_area_corrupted": round(df_corrupt["area"].mean(), 1) if len(df_corrupt) else 0.0,
            "gate_passed_clean": passed_clean,
            "gate_passed_corrupted": passed_corrupt,
            "gate_reason_corrupted": reason_corrupt,
            "gate_caught_it": not passed_corrupt,  # False = corruption slipped through undetected
        })
    return pd.DataFrame(rows)




def unet_vs_otsu_comparison(unet_model: UNet, image_dir: str | Path, mask_dir: str | Path) -> pd.DataFrame:
    """For every image in `image_dir`, segment it both with the trained
    U-Net and with Otsu+morphology, score both against the SAME
    ground-truth mask (Dice/IoU), and return one row per image so the
    best/worst examples for each method can be picked directly from the
    table rather than eyeballed."""
    from .data_prep import load_and_preprocess

    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    rows = []
    for p in sorted(image_dir.glob("*.png")):
        gray = load_and_preprocess(p)
        gt = np.asarray(load_and_preprocess(mask_dir / p.name) > 0.5)

        unet_mask = unet_segment(unet_model, gray)
        otsu_mask = otsu_segment(gray)

        dice_unet, iou_unet = mask_dice(unet_mask, gt), mask_iou(unet_mask, gt)
        dice_otsu, iou_otsu = mask_dice(otsu_mask, gt), mask_iou(otsu_mask, gt)

        rows.append({
            "image_id": p.stem,
            "dice_unet": dice_unet, "iou_unet": iou_unet,
            "dice_otsu": dice_otsu, "iou_otsu": iou_otsu,
            "dice_diff_unet_minus_otsu": dice_unet - dice_otsu,
        })
    return pd.DataFrame(rows).sort_values("dice_diff_unet_minus_otsu", ascending=False).reset_index(drop=True)
