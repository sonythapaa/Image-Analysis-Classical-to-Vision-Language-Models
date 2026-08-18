"""End-to-end pipeline for the assignment.

    python scripts/run_pipeline.py --stage all

Stages
------
eda        Task 1: preprocessing + EDA figures.
vlm        Task 1: naive vs structured prompt on a sample image.

classical  Task 2: Otsu + regionprops on a sample image (deterministic part).
llm2       Task 2: numbers-first description. Needs Ollama.

train      Task 3: train the U-Net from scratch and evaluate on the test set.
loss_ablation
           Task 3 extension: trains BCE-only, Dice-only, and BCE+Dice from
           scratch and compares final test Dice/IoU. Needs `train` to have
           been run at least once first isn't required — this trains its
           own fresh models — but is slower (trains 3 models, not 1).
instance_and_robustness
           comparing ground-truth vs regionprops-on-predicted-mask object
           counts, and runs the corruption-tracing extension using
           data/test_corrupted/. Needs `train` to have been run first (uses
           the saved U-Net weights).

hybrid     Task 4: full pipeline on every unseen test image -> CSV.
           The deterministic stages (U-Net + regionprops + quality gate)
           
all        Run every stage above, in order.

Run this from the project root (the directory containing `data/`, `src/`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nuclei_pipeline.data_prep import (DatasetPaths, load_metadata, preprocess_split,
                                        intensity_histogram_data,
                                        ensure_noise_corruption)
from nuclei_pipeline.classical_features import otsu_segment, region_feature_table, features_to_text
from nuclei_pipeline.dataset import NucleiDataset
from nuclei_pipeline.unet import UNet
from nuclei_pipeline.train import train_model, run_epoch, loss_ablation, dice_coefficient, iou_score
from nuclei_pipeline import llm_vision
from nuclei_pipeline.hybrid_pipeline import (load_unet, run_batch, full_pipeline,
                                              instance_merging_analysis, robustness_trace,
                                              unet_vs_otsu_comparison,
                                              unet_segment, measure, summarise, quality_gate)
from nuclei_pipeline.classical_features import otsu_segment
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)


def stage_eda():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = DatasetPaths(DATA)
    meta = load_metadata(paths)

    # train-split-only EDA (the split used to fit the model)
    train_imgs = preprocess_split(paths, "train")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, iid in zip(axes.ravel(), sorted(train_imgs)[:8]):
        ax.imshow(train_imgs[iid], cmap="gray")
        row = meta[meta.image_id == iid].iloc[0]
        ax.set_title(f"{iid}\n{row['density']}, n={row['n_objects']}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "eda_sample_grid.png", dpi=150)
    plt.close(fig)

    counts, edges = intensity_histogram_data(train_imgs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(edges[:-1], counts, width=(edges[1] - edges[0]), align="edge")
    ax.set_xlabel("Pixel intensity (grayscale, 0-1)")
    ax.set_ylabel("Pixel count")
    ax.set_title("Pooled intensity histogram — train split (n=80 images)")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "eda_intensity_histogram.png", dpi=150)
    plt.close(fig)



def stage_vlm(sample_image: str = "train_004"):
    img_path = DATA / "train" / "images" / f"{sample_image}.png"
    try:
        result = llm_vision.compare_naive_vs_structured(img_path)
        variability = llm_vision.run_variability_check(img_path, llm_vision.STRUCTURED_PROMPT, n=3)
    except RuntimeError as e:
        print(f"[vlm] SKIPPED — {e}")
        return
    result["variability_check_n3"] = variability
    result["variability_all_identical"] = len(set(variability)) == 1
    (OUT / "tables" / "vlm_comparison.json").write_text(json.dumps(result, indent=2))
    print("[vlm] saved outputs/tables/vlm_comparison.json")
    print(f"[vlm] variability check: {len(set(variability))} distinct outputs out of 3 runs")


def stage_classical(sample_image: str = "train_004"):
    from nuclei_pipeline.data_prep import load_and_preprocess
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_path = DATA / "train" / "images" / f"{sample_image}.png"
    gray = load_and_preprocess(img_path)
    mask = otsu_segment(gray)
    df = region_feature_table(gray, mask)
    df.to_csv(OUT / "tables" / "feature_table.csv", index=False)
    print("[classical]", features_to_text(df))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Grayscale input"); axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Otsu + morphology mask"); axes[1].axis("off")
    axes[2].hist(df["area"], bins=20); axes[2].set_title("Object area distribution"); axes[2].set_xlabel("area (px)")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "otsu_example.png", dpi=150)
    plt.close(fig)
    print("[classical] saved figure + feature table")


def stage_llm2(sample_image: str = "train_004"):
    from nuclei_pipeline.data_prep import load_and_preprocess
    from nuclei_pipeline.classical_features import numbers_first_description

    img_path = DATA / "train" / "images" / f"{sample_image}.png"
    gray = load_and_preprocess(img_path)
    mask = otsu_segment(gray)
    df = region_feature_table(gray, mask)
    try:
        result = numbers_first_description(df)
    except RuntimeError as e:
        print(f"[llm2] SKIPPED — {e}")
        return
    (OUT / "tables" / "llm_interpretation.json").write_text(json.dumps(result, indent=2))
    print("[llm2] saved outputs/tables/llm_interpretation.json")


def stage_train(epochs: int = 15, lr: float = 1e-3, loss_name: str = "bce_dice"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cpu")
    np.random.seed(42); torch.manual_seed(42)

    train_ds = NucleiDataset(DATA, "train", augment=True)
    val_ds = NucleiDataset(DATA, "val", augment=False)
    test_ds = NucleiDataset(DATA, "test", augment=False)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = UNet(base=16).to(device)
    history = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, loss_name=loss_name)
    torch.save(model.state_dict(), OUT / "models" / "unet_nuclei_final.pth")


    val_metrics = run_epoch(model, val_loader, "bce_dice", device, optimizer=None)
    (OUT / "tables" / "validation_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    print("[train] final validation metrics :", val_metrics)

    test_metrics = run_epoch(model, test_loader, "bce_dice", device, optimizer=None)
    (OUT / "tables" / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (OUT / "tables" / "history.json").write_text(json.dumps(history, indent=2))
    print("[train] test metrics (unseen test split):", test_metrics)

    # --- training curves (loss + Dice/IoU) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"Loss ({loss_name})"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["train_dice"], label="train dice")
    axes[1].plot(history["val_dice"], label="val dice")
    axes[1].plot(history["val_iou"], label="val iou", linestyle="--")
    axes[1].set_title("Dice / IoU"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "training_curves.png", dpi=150)
    plt.close(fig)

    # --- input / ground-truth / prediction panels for 3 validation images ---
    model.eval()
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    with torch.no_grad():
        for i in range(3):
            x, y = val_ds[i]
            logits = model(x.unsqueeze(0))
            pred = (torch.sigmoid(logits) > 0.5).float()[0, 0]
            d = dice_coefficient(logits, y.unsqueeze(0))
            iou = iou_score(logits, y.unsqueeze(0))
            axes[i, 0].imshow(x[0], cmap="gray"); axes[i, 0].set_title("Input"); axes[i, 0].axis("off")
            axes[i, 1].imshow(y[0], cmap="gray"); axes[i, 1].set_title("Ground truth"); axes[i, 1].axis("off")
            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title(f"U-Net prediction (d={d:.3f}, iou={iou:.3f})", fontsize=9)
            axes[i, 2].axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "predictions.png", dpi=150)
    plt.close(fig)

    # --- per-image test-set Dice/IoU, so we can point at specific failure cases ---
    rows = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            logits = model(x.unsqueeze(0))
            rows.append({
                "image_id": test_ds.image_id(i),
                "dice": dice_coefficient(logits, y.unsqueeze(0)),
                "iou": iou_score(logits, y.unsqueeze(0)),
            })
    pd.DataFrame(rows).sort_values("dice").to_csv(OUT / "tables" / "per_image_test_metrics.csv", index=False)
    print("[train] saved training curves, prediction panels, per-image test metrics")
    return model  # returned so other stages (loss_ablation excluded) can reuse the trained weights in-process


def stage_loss_ablation(epochs: int = 4, lr: float = 1e-3):
    """ BCE vs Dice vs BCE+Dice, same data/seed/epochs for
    a fair comparison. Trains three fresh models from scratch — this is
    slower than the other stages (~3x a single training run).
    raise it for a more thorough (but slower) comparison."""
    device = torch.device("cpu")
    train_ds = NucleiDataset(DATA, "train", augment=True)
    val_ds = NucleiDataset(DATA, "val", augment=False)
    test_ds = NucleiDataset(DATA, "test", augment=False)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    results = loss_ablation(lambda: UNet(base=16), train_loader, val_loader, test_loader,
                             device, epochs=epochs, lr=lr)
    (OUT / "tables" / "loss_ablation.json").write_text(json.dumps(results, indent=2))
    print("[loss_ablation] saved outputs/tables/loss_ablation.json")


def stage_instance_and_robustness():
    """the robustness extension.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cpu")
    weights = OUT / "models" / "unet_nuclei_final.pth"
    if not weights.exists():
        print("[instance_and_robustness] SKIPPED — run --stage train first (no trained weights found).")
        return
    model = load_unet(weights, device)
    meta = load_metadata(DatasetPaths(DATA))

    # --- instance-merging analysis: pixel Dice vs object-count undercount ---
    merge_df = instance_merging_analysis(model, meta, DATA / "test" / "images")
    merge_df.to_csv(OUT / "tables" / "instance_merging_analysis.csv", index=False)
    print("[instance_and_robustness] instance-merging analysis:")
    print(merge_df.to_string(index=False))

    # --- robustness extension: trace test_corrupted/ images through the pipeline ---
    ensure_noise_corruption(DatasetPaths(DATA))  # adds the synthetic-noise variant (3rd corruption type)
    corrupted_dir = DATA / "test_corrupted" / "images"
    if corrupted_dir.exists() and any(corrupted_dir.glob("*.png")):
        rob_df = robustness_trace(model, corrupted_dir, DATA / "test" / "images")
        rob_df.to_csv(OUT / "tables" / "task_extension_robustness.csv", index=False)

  
        from nuclei_pipeline.data_prep import load_and_preprocess
        base_ids = sorted(rob_df["base_image"].unique())
        variants = sorted(rob_df["variant"].unique())  # e.g. ["blur", "lowcontrast"]
        n_cols = 1 + 2 * len(variants)  # original + (corrupted input, mask) per variant

        fig, axes = plt.subplots(len(base_ids), n_cols, figsize=(3.2 * n_cols, 3.4 * len(base_ids)))
        if len(base_ids) == 1:
            axes = axes.reshape(1, -1)

        for row, base_id in enumerate(base_ids):
            clean_path = DATA / "test" / "images" / f"{base_id}.png"
            gray_clean = load_and_preprocess(clean_path)
            mask_clean = unet_segment(model, gray_clean)
            n_clean = len(measure(gray_clean, mask_clean))
            axes[row, 0].imshow(gray_clean, cmap="gray")
            axes[row, 0].set_title(f"{base_id}\noriginal (n={n_clean})", fontsize=9, pad=12)
            axes[row, 0].axis("off")

            for v_idx, variant in enumerate(variants):
                cf = corrupted_dir / f"{base_id}_{variant}.png"
                gray_corrupt = load_and_preprocess(cf)
                mask_corrupt = unet_segment(model, gray_corrupt)
                n_obj = len(measure(gray_corrupt, mask_corrupt))
                col_input = 1 + 2 * v_idx
                col_mask = col_input + 1
                axes[row, col_input].imshow(gray_corrupt, cmap="gray")
                axes[row, col_input].set_title(f"{variant} input", fontsize=9)
                axes[row, col_input].axis("off")
                axes[row, col_mask].imshow(mask_corrupt, cmap="gray")
                axes[row, col_mask].set_title(f"U-Net mask, n={n_obj}", fontsize=9)
                axes[row, col_mask].axis("off")

        plt.tight_layout()
        fig.subplots_adjust(hspace=0.45)
        fig.savefig(OUT / "figures" / "task_extension_robustness.png", dpi=150)
        plt.close(fig)
        print("[instance_and_robustness] robustness trace:")
        print(rob_df.to_string(index=False))
    else:
        print("[instance_and_robustness] no data/test_corrupted/images found — skipping robustness extension")


def stage_unet_vs_otsu():
    """per-test-image U-Net vs Otsu
    comparison against the same ground truth. Needs --stage train run
    first (uses the saved U-Net weights)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nuclei_pipeline.data_prep import load_and_preprocess

    device = torch.device("cpu")
    weights = OUT / "models" / "unet_nuclei_final.pth"
    if not weights.exists():
        print("[unet_vs_otsu] SKIPPED — run --stage train first (no trained weights found).")
        return
    model = load_unet(weights, device)

    df = unet_vs_otsu_comparison(model, DATA / "test" / "images", DATA / "test" / "masks")
    df.to_csv(OUT / "tables" / "test_unet_vs_otsu.csv", index=False)
    print("[unet_vs_otsu] per-image comparison:")
    print(df.to_string(index=False))

    # Report-ready mean summary
    summary = pd.DataFrame({
        "method": ["U-Net", "Otsu"],
        "mean_dice": [df["dice_unet"].mean(), df["dice_otsu"].mean()],
        "mean_iou": [df["iou_unet"].mean(), df["iou_otsu"].mean()],
    })
    summary.to_csv(OUT / "tables" / "test_metrics_summary.csv", index=False)
    print("[unet_vs_otsu] summary:")
    print(summary.to_string(index=False))

    # Bar chart: per-image Dice, U-Net vs Otsu, sorted by U-Net's advantage
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["dice_unet"], width=0.4, label="U-Net")
    ax.bar([i + 0.2 for i in x], df["dice_otsu"], width=0.4, label="Otsu")
    ax.set_xticks(list(x)); ax.set_xticklabels(df["image_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Dice"); ax.set_title("U-Net vs Otsu — per-image Dice (test set)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "unet_vs_otsu.png", dpi=150)
    plt.close(fig)

    # Example panels: the image where U-Net helps most, and where it helps least
    # (or where Otsu actually wins, if any row has a negative diff)
    best_row = df.iloc[0]    # largest positive dice_diff -> U-Net's biggest win
    worst_row = df.iloc[-1]  # smallest (or negative) diff -> U-Net's weakest showing / Otsu's best case

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for row_idx, row in enumerate([best_row, worst_row]):
        gray = load_and_preprocess(DATA / "test" / "images" / f"{row['image_id']}.png")
        gt = load_and_preprocess(DATA / "test" / "masks" / f"{row['image_id']}.png") > 0.5
        unet_mask = unet_segment(model, gray)
        otsu_mask = otsu_segment(gray)

        label = "Largest U-Net advantage" if row_idx == 0 else "Smallest U-Net advantage"
        axes[row_idx, 0].imshow(gray, cmap="gray")
        axes[row_idx, 0].set_title(f"{row['image_id']}\n{label}", fontsize=9)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(gt, cmap="gray"); axes[row_idx, 1].set_title("Ground truth"); axes[row_idx, 1].axis("off")
        axes[row_idx, 2].imshow(otsu_mask, cmap="gray")
        axes[row_idx, 2].set_title(f"Otsu (dice={row['dice_otsu']:.3f})", fontsize=9); axes[row_idx, 2].axis("off")
        axes[row_idx, 3].imshow(unet_mask, cmap="gray")
        axes[row_idx, 3].set_title(f"U-Net (dice={row['dice_unet']:.3f})", fontsize=9); axes[row_idx, 3].axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "method_example_panels.png", dpi=150)
    plt.close(fig)
    print("[unet_vs_otsu] saved comparison CSV, summary CSV, bar chart, and example panels")


def stage_export_prompts():
    """Consolidates every LLM prompt used anywhere in the pipeline into one
    file, and by anyone trying to reproduce this exactly."""
    from nuclei_pipeline.classical_features import NUMBERS_FIRST_PROMPT_TEMPLATE
    from nuclei_pipeline.hybrid_pipeline import HYBRID_PROMPT_TEMPLATE

    prompts = {
        "task1_naive_prompt": llm_vision.NAIVE_PROMPT,
        "task1_structured_prompt": llm_vision.STRUCTURED_PROMPT,
        "task1_vlm_model": llm_vision.MODEL,
        "task2_numbers_first_prompt_template": NUMBERS_FIRST_PROMPT_TEMPLATE,
        "task2_text_model": "llama3.2",
        "task4_hybrid_prompt_template": HYBRID_PROMPT_TEMPLATE,
        "task4_text_model": "llama3.2",
    }
    (OUT / "tables" / "prompts_used.json").write_text(json.dumps(prompts, indent=2))
    print("[export_prompts] saved outputs/tables/prompts_used.json")


def stage_hybrid():
    device = torch.device("cpu")
    weights = OUT / "models" / "unet_nuclei_final.pth"
    if not weights.exists():
        print("[hybrid] SKIPPED — run the 'train' stage first (no trained U-Net weights found).")
        return
    model = load_unet(weights, device)

    try:
        df = run_batch(DATA / "test" / "images", model)
        df.to_csv(OUT / "tables" / "hybrid_report.csv", index=False)
        print(f"[hybrid] saved outputs/tables/hybrid_report.csv ({len(df)} rows)")
    except RuntimeError as e:
        print(f"[hybrid] LLM stage SKIPPED — {e}")
        print("[hybrid] Falling back to the deterministic stages only (U-Net + regionprops + gate).")
        from nuclei_pipeline.hybrid_pipeline import unet_segment, measure, summarise, quality_gate
        from nuclei_pipeline.data_prep import load_and_preprocess
        rows = []
        for p in sorted((DATA / "test" / "images").glob("*.png")):
            gray = load_and_preprocess(p)
            mask = unet_segment(model, gray)
            feat = measure(gray, mask)
            passed, reason = quality_gate(feat)
            rows.append({"image_id": p.stem, "n_objects_measured": len(feat),
                         "mean_area_measured": round(feat["area"].mean(), 1) if len(feat) else 0.0,
                         "gate_passed": passed, "gate_reason": reason,
                         "summary_text": summarise(feat, p.stem)})
        pd.DataFrame(rows).to_csv(OUT / "tables" / "deterministic_only.csv", index=False)
        print("[hybrid] saved outputs/tables/deterministic_only.csv (LLM columns pending)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all",
                         choices=["eda", "vlm", "classical", "llm2", "train", "loss_ablation",
                                  "instance_and_robustness", "unet_vs_otsu", "export_prompts", "hybrid", "all"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    stages = {
        "eda": stage_eda, "vlm": stage_vlm, "classical": stage_classical,
        "llm2": stage_llm2, "train": lambda: stage_train(epochs=args.epochs),
        "loss_ablation": stage_loss_ablation, "instance_and_robustness": stage_instance_and_robustness,
        "unet_vs_otsu": stage_unet_vs_otsu, "export_prompts": stage_export_prompts, "hybrid": stage_hybrid,
    }
    if args.stage == "all":
        for name, fn in stages.items():
            print(f"\n Stage: {name}")
            fn()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
