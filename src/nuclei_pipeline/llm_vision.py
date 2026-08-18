"""Send a representative image to a local multimodal
model (qwen2.5vl) via Ollama, comparing a naive prompt against a
structured, descriptive-not-diagnostic prompt that forces valid JSON.

NOTE — this module needs a running local Ollama server with
`qwen2.5vl` pulled (`ollama pull qwen2.5vl`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

try:
    from ollama import chat
except ImportError:  # pragma: no cover - only hit if ollama isn't installed
    chat = None

import os

# NOTE ON MODEL SUBSTITUTION: the assignment brief names llama3.2-vision.
# On this project's development/test machine, llama3.2-vision fails to load
# under current Ollama versions (>=0.30.0) with "unknown model architecture:
# 'mllama'" — a confirmed, currently-open upstream regression (see
# github.com/ollama/ollama/issues/16547, closed as duplicate of #16490):
# Ollama's rewritten engine dropped mllama support, which was previously
# only available via Ollama's private llama.cpp fork and was never in
# upstream llama.cpp. qwen2.5vl is used instead as a like-for-like local
# multimodal substitute; override via the VLM_MODEL environment variable.
MODEL = os.environ.get("VLM_MODEL", "qwen2.5vl")


# Prompts
NAIVE_PROMPT = "Describe this image."

STRUCTURED_PROMPT = """You are assisting with an EDUCATIONAL image-description
exercise. You are NOT a clinician and this is NOT a diagnostic exercise.

Describe what is visually present in this microscopy image. Do not diagnose,
do not name a disease, and do not guess at anything you cannot see directly
in the pixels (e.g. never guess patient age, sex, or clinical history).

If you are unsure about any attribute, write "uncertain" for that field
rather than guessing.

Return your answer as a single valid JSON object with EXACTLY these keys,
and nothing else before or after the JSON:
{
  "modality": "<imaging modality you can infer, or 'uncertain'>",
  "tissue_type": "<what kind of tissue/sample this looks like, or 'uncertain'>",
  "notable_features": "<one sentence on what stands out visually>",
  "image_quality": "<one of: good, fair, poor, uncertain>"
}
"""


def save_array_as_png(arr, path: str | Path) -> Path:
    """Save a float [0,1] or uint8 array as a PNG Ollama can read."""
    path = Path(path)
    if arr.dtype != "uint8":
        arr = (arr * 255).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(path)
    return path


def describe_image(image_path: str | Path, prompt: str, temperature: float = 0.0) -> str:
    """Single multimodal call. Returns the raw model text."""
    if chat is None:
        raise RuntimeError(
            "The `ollama` package / server isn't available in this environment. "
            f"Run this on a machine with Ollama installed and `{MODEL}` pulled."
        )
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def parse_model_json(text: str) -> dict | None:
    """Extract the first {...} block from a model reply and parse it as JSON.

    Models occasionally wrap JSON in prose or markdown fences despite
    instructions, so we search for the first balanced-looking brace block
    rather than assuming `text` is pure JSON.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def run_variability_check(image_path: str | Path, prompt: str, n: int = 3, temperature: float = 0.8) -> list[str]:
    """Run the same prompt n times at temperature>0 to demonstrate that
    repeated runs are not identical (Task 1 requirement)."""
    return [describe_image(image_path, prompt, temperature=temperature) for _ in range(n)]


def compare_naive_vs_structured(image_path: str | Path) -> dict:
    """Run both prompts once each (temperature=0 for the structured one, so
    the JSON is reproducible) and return everything needed for the report."""
    naive = describe_image(image_path, NAIVE_PROMPT, temperature=0.7)
    structured_raw = describe_image(image_path, STRUCTURED_PROMPT, temperature=0.0)
    structured_json = parse_model_json(structured_raw)
    return {
        "naive_prompt": NAIVE_PROMPT,
        "naive_output": naive,
        "structured_prompt": STRUCTURED_PROMPT,
        "structured_output_raw": structured_raw,
        "structured_output_json": structured_json,
    }
