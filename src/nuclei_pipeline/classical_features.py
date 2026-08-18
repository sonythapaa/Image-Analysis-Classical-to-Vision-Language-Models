"""classical (non-learned) segmentation and feature extraction —
Otsu thresholding, morphological cleanup, connected-component labelling,
and a regionprops feature table — followed by a numbers-only LLM summary.

The LLM never sees the image in this task: only the text summary derived
from the feature table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, opening, closing, disk
from skimage.measure import label, regionprops_table

try:
    from ollama import chat
except ImportError: 
    chat = None

MODEL = "llama3.2"  # text-only model is enough once the LLM only sees numbers

PROPERTIES = ("label", "area", "eccentricity", "solidity", "mean_intensity",
              "perimeter", "equivalent_diameter_area")


def mask_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Dice coefficient between two boolean masks (numpy arrays) — the
    array-based counterpart to train.py's dice_coefficient(), which expects
    torch logits. Used to score Otsu's output, which is never a tensor."""
    pred_mask, gt_mask = pred_mask.astype(bool), gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    return (2 * intersection + eps) / (total + eps)


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """IoU between two boolean masks (numpy arrays)."""
    pred_mask, gt_mask = pred_mask.astype(bool), gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return (intersection + eps) / (union + eps)


def otsu_segment(gray: np.ndarray, min_size: int = 15) -> np.ndarray:
    """Otsu threshold + morphological opening/closing to clean up noise.

    `gray` is a float array in [0, 1]. Returns a boolean foreground mask.
    """
    thresh = threshold_otsu(gray)
    mask = gray > thresh
    mask = opening(mask, disk(1))
    mask = closing(mask, disk(1))
    mask = remove_small_objects(mask, min_size=min_size)
    return mask


def region_feature_table(gray: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Label connected components and compute a regionprops feature table."""
    labels = label(mask)
    if labels.max() == 0:
        return pd.DataFrame(columns=PROPERTIES)
    table = regionprops_table(labels, intensity_image=gray, properties=PROPERTIES)
    return pd.DataFrame(table)


def features_to_text(df: pd.DataFrame) -> str:
    """Turn a regionprops feature table into a short factual sentence — the
    ONLY thing the LLM in this task is allowed to see (numbers, no image)."""
    if len(df) == 0:
        return "No objects were detected above the size threshold."
    n = len(df)
    mean_area = df["area"].mean()
    mean_ecc = df["eccentricity"].mean()
    mean_solidity = df["solidity"].mean()
    mean_intensity = df["mean_intensity"].mean()
    cv_area = df["area"].std() / mean_area if mean_area > 0 else 0.0
    return (
        f"Detected {n} objects. Mean area {mean_area:.1f} px "
        f"(coefficient of variation {cv_area:.2f}). Mean eccentricity "
        f"{mean_ecc:.2f} (0=circular, 1=elongated). Mean solidity "
        f"{mean_solidity:.2f} (1=fully convex). Mean intensity {mean_intensity:.2f}."
    )


NUMBERS_FIRST_PROMPT_TEMPLATE = """You are given ONLY numeric measurements
from an automated image-analysis step. You have NOT seen the image itself.

Measurements:
{summary_text}

Write one paragraph describing what these numbers suggest about the scene,
staying strictly within what the numbers support. Then return a second,
separate JSON object with EXACTLY these keys:
{{
  "n_objects": <integer>,
  "density_class": "<one of: sparse, normal, dense, uncertain>",
  "shape_regularity": "<one of: round/regular, elongated/irregular, uncertain>",
  "quality_flag": "<one of: ok, low_count, uncertain>"
}}
"""


def numbers_first_description(feature_table: pd.DataFrame, temperature: float = 0.0) -> dict:
    if chat is None:
        raise RuntimeError(
            "The `ollama` package / server isn't available in this environment. "
            "Run this where Ollama is installed."
        )
    summary_text = features_to_text(feature_table)
    prompt = NUMBERS_FIRST_PROMPT_TEMPLATE.format(summary_text=summary_text)
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return {"prompt": prompt, "summary_text": summary_text, "output_raw": response["message"]["content"]}
