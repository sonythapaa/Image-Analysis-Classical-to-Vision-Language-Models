# Hybrid Biomedical Image-Analysis Pipeline — Nuclei Segmentation

Assignment: multimodal LLM description + classical image processing + U-Net
segmentation + hybrid pipeline, applied to a synthetic nuclei
fluorescence-microscopy dataset.

Pipeline: **raw image → segmentation → quantitative region features →
structured JSON record → short narrative.**

## Dataset

`data/` is the dataset from
https://github.com/Nickolay-K/Assingnment-3-dataset — downloaded as-is via
`codeload.github.com/Nickolay-K/Assingnment-3-dataset/zip/refs/heads/main`
and unzipped, with no modification. Its own `README.md`,
`dataset_summary.json`, and `make_dataset.py` (the seeded generator script
the repo owner used to create it) are preserved unchanged inside `data/` as
provenance.

**Note on splits**: the GitHub repo does not ship a separate unsplit "raw"
version — its own `make_dataset.py` generates images directly into
`train/` (80), `val/` (20), and `test/` (12) with different random seeds
per split.

```
data/
  README.md, dataset_summary.json, make_dataset.py   <- straight from the GitHub repo
  metadata.csv                                        <- ground truth for every image, all splits
  train/  images/ masks/ labels/   (80 pairs)          <- from the GitHub repo
  val/    images/ masks/ labels/   (20 pairs)          <- from the GitHub repo
  test/   images/ masks/ labels/   (12 pairs)          <- from the GitHub repo
  test_corrupted/images/           (4 files)           <- from the GitHub repo
```

```
data/
  train/  images/ masks/ labels/   (80 pairs)
  val/    images/ masks/ labels/   (20 pairs)
  test/   images/ masks/ labels/   (12 pairs — "unseen" for Tasks 3 and 4)
  test_corrupted/images/           (blur / low-contrast variants, for the robustness extension)
  metadata.csv                     (ground-truth n_objects, density class, etc. per image)
```
Images are 256×256 RGB, DAPI-like blue-stained nuclei on a dark field.
Masks are binary PNGs (0/255); `labels/` holds 16-bit per-nucleus instance
IDs if you want to do instance-level work beyond what's required here.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

For Tasks 1, 2, and 4's LLM steps you also need
[Ollama](https://ollama.com) installed and running locally, with the models
pulled:
```bash
ollama pull qwen2.5vl        # vision model 
ollama pull llama3.2         # text model 
```
Start the server (`ollama serve`) before running any `vlm`/`llm2`/`hybrid`
pipeline stage.

**Model substitution note**: the assignment brief names `llama3.2-vision`
for Task 1(Vision Task). `llama3.2-vision` fails to load
under Ollama >=0.30.0 with `unknown model architecture: 'mllama'`.
`qwen2.5vl` is used as a like-for-like local multimodal substitute.
If `llama3.2-vision` does run on your machine, set
`$env:VLM_MODEL = "llama3.2-vision"` before running `--stage vlm` to use it
instead.

## Project layout

```
src/nuclei_pipeline/
  data_prep.py          Task 1: loading, grayscale, resize, EDA, ensure_combined_view()
  llm_vision.py         Task 1: naive vs structured-JSON VLM prompt, variability check
  classical_features.py Task 2: Otsu + morphology + regionprops + numbers-first LLM prompt
  dataset.py             PyTorch Dataset for the real train/val/test split
  unet.py                Task 3: U-Net architecture
  train.py                Task 3: losses (BCE / Dice / BCE+Dice), Dice & IoU metrics, training loop, loss_ablation()
  hybrid_pipeline.py     Task 4: U-Net mask -> regionprops -> quality gate -> LLM JSON + narrative;
                          also instance_merging_analysis() and robustness_trace() (Task 3/extension discussion)
scripts/run_pipeline.py  Runs every stage; run with --stage {eda,vlm,classical,llm2,train,
                          loss_ablation,instance_and_robustness,hybrid,all}
tests/                   pytest unit tests for every deterministic function (25 tests, all passing)
outputs/
  figures/               EDA grid, intensity histogram, Otsu example, training curves, prediction panels
  tables/                feature tables, test metrics, loss ablation, hybrid CSV report
  models/                trained U-Net weights (unet_nuclei_final.pth)
```

## Running it

```bash
# Everything except the LLM-dependent steps (safe without Ollama running):
python scripts/run_pipeline.py --stage eda
python scripts/run_pipeline.py --stage classical
python scripts/run_pipeline.py --stage train --epochs 15
python scripts/run_pipeline.py --stage loss_ablation
python scripts/run_pipeline.py --stage instance_and_robustness   # needs --stage train run first

# LLM-dependent steps (need Ollama running):
python scripts/run_pipeline.py --stage vlm
python scripts/run_pipeline.py --stage llm2
python scripts/run_pipeline.py --stage hybrid

# Or all at once:
python scripts/run_pipeline.py --stage all --epochs 15

# Tests:
pytest tests/ -v
```

