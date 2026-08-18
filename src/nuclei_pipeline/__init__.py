"""nuclei_pipeline — hybrid biomedical image-analysis pipeline for the
synthetic DAPI-stained nuclei dataset (fluorescence microscopy modality).

Modules
-------
data_prep          : loading, grayscale conversion, resizing, EDA
llm_vision          : Task 1 — multimodal (VLM) description via Ollama
classical_features  : Task 2 — Otsu + regionprops + numbers-first LLM summary
dataset             : PyTorch Dataset for the real nuclei train/val/test split
unet                : Task 3 — U-Net architecture (matches Lab 4)
train               : Task 3 — training loop, losses, Dice/IoU metrics
hybrid_pipeline     : Task 4 — U-Net mask -> regionprops -> LLM JSON -> narrative
"""
