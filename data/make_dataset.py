"""
make_dataset.py
----------------
Generate a small, realistic *stained-nuclei* fluorescence-microscopy dataset
with exact ground-truth instance/binary masks, in the same spirit as the
2018 Data Science Bowl nuclei set (images + masks), but fully synthetic so
that (a) it can be distributed without any licensing constraints and
(b) the ground truth is known exactly, which is what an answer key needs.

Each image is 256x256 RGB (blue-stained nuclei on a dark field, DAPI-like).
For every image we also save:
  - a binary mask  (255 = nucleus, 0 = background)
  - a label mask   (instance labels, 16-bit) for optional instance work
Ground-truth per-image statistics are written to metadata.csv.

Splits:  train (80) / val (20) / test (12, images only used as "unseen")
Plus a handful of deliberately corrupted test images for the robustness part.
"""
import os, csv, json
import numpy as np
from skimage.draw import ellipse
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.io import imsave
import imageio.v2 as imageio

RNG_SEED = 20260715
IMG = 256

def _rng(seed):
    return np.random.default_rng(seed)

def make_one(seed, density="normal"):
    """Return (rgb_uint8, binary_uint8, label_uint16, stats dict)."""
    rng = _rng(seed)
    label = np.zeros((IMG, IMG), np.uint16)
    intensity = np.zeros((IMG, IMG), np.float32)

    # choose number of nuclei by density regime
    n_map = {"sparse": (4, 12), "normal": (15, 40), "dense": (45, 85), "clustered": (30, 60)}
    lo, hi = n_map[density]
    n_target = int(rng.integers(lo, hi + 1))

    placed = 0
    attempts = 0
    centers = []
    while placed < n_target and attempts < n_target * 25:
        attempts += 1
        if density == "clustered" and centers and rng.random() < 0.6:
            # place near an existing centre to form clumps
            cx0, cy0 = centers[rng.integers(0, len(centers))]
            r = rng.normal(0, 10)
            cc = int(np.clip(cx0 + r, 8, IMG - 8))
            rr = int(np.clip(cy0 + rng.normal(0, 10), 8, IMG - 8))
        else:
            rr = int(rng.integers(8, IMG - 8))
            cc = int(rng.integers(8, IMG - 8))
        # nucleus size / shape
        a = rng.uniform(5, 12)              # semi-major
        b = a * rng.uniform(0.55, 0.98)     # semi-minor -> eccentricity variety
        theta = rng.uniform(0, np.pi)
        ry, rx = ellipse(rr, cc, a, b, rotation=theta, shape=(IMG, IMG))
        if ry.size == 0:
            continue
        placed += 1
        centers.append((cc, rr))
        label[ry, rx] = placed
        # textured intensity: bright core, softer edge, per-nucleus brightness
        core = rng.uniform(0.55, 1.0)
        intensity[ry, rx] = np.maximum(intensity[ry, rx], core)

    # soft blur to mimic microscope PSF, then add nucleoli speckle
    intensity = gaussian(intensity, sigma=1.1)
    speck = rng.random((IMG, IMG)) < 0.002
    intensity[speck] = np.minimum(1.0, intensity[speck] + 0.3)

    # background haze + poisson-like noise
    bg = rng.uniform(0.02, 0.06)
    intensity = np.clip(intensity + bg, 0, 1)
    intensity = random_noise(intensity, mode="gaussian", var=0.0015, rng=rng)
    intensity = np.clip(intensity, 0, 1)

    # DAPI-like blue staining: put signal mostly in blue, a little in green
    rgb = np.zeros((IMG, IMG, 3), np.float32)
    rgb[..., 2] = intensity                      # blue
    rgb[..., 1] = intensity * 0.35               # green
    rgb[..., 0] = intensity * 0.12               # red
    rgb = np.clip(rgb, 0, 1)
    rgb_u8 = (rgb * 255).astype(np.uint8)

    binary = (label > 0).astype(np.uint8) * 255

    stats = dict(n_objects=int(placed), density=density,
                 mean_intensity=float(intensity[label > 0].mean()) if placed else 0.0,
                 area_fraction=float((label > 0).mean()))
    return rgb_u8, binary, label.astype(np.uint16), stats


def build(out_dir="nuclei_dataset"):
    layout = {
        "train": (80, RNG_SEED),
        "val":   (20, RNG_SEED + 10_000),
        "test":  (12, RNG_SEED + 20_000),
    }
    densities = ["sparse", "normal", "normal", "normal", "dense", "clustered"]
    meta_rows = []
    for split, (n, base) in layout.items():
        for sub in ["images", "masks", "labels"]:
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)
        for i in range(n):
            seed = base + i
            dens = densities[i % len(densities)]
            rgb, binmask, lab, st = make_one(seed, dens)
            stem = f"{split}_{i:03d}"
            imsave(os.path.join(out_dir, split, "images", stem + ".png"), rgb, check_contrast=False)
            imsave(os.path.join(out_dir, split, "masks", stem + ".png"), binmask, check_contrast=False)
            imageio.imwrite(os.path.join(out_dir, split, "labels", stem + ".png"), lab)
            st.update(split=split, image_id=stem, seed=int(seed))
            meta_rows.append(st)

    # corrupted variants of two test images (for the robustness part)
    corr_dir = os.path.join(out_dir, "test_corrupted", "images")
    os.makedirs(corr_dir, exist_ok=True)
    from skimage.filters import gaussian as gb
    for src in ["test_000", "test_004"]:
        rgb = imageio.imread(os.path.join(out_dir, "test", "images", src + ".png")).astype(np.float32) / 255
        blur = np.clip(gb(rgb, sigma=6, channel_axis=-1), 0, 1)
        imsave(os.path.join(corr_dir, src + "_blur.png"), (blur * 255).astype(np.uint8), check_contrast=False)
        contrast = np.clip((rgb - 0.5) * 0.15 + 0.5, 0, 1)   # crushed contrast
        imsave(os.path.join(corr_dir, src + "_lowcontrast.png"), (contrast * 255).astype(np.uint8), check_contrast=False)

    # metadata.csv
    with open(os.path.join(out_dir, "metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "split", "density", "n_objects",
                                          "mean_intensity", "area_fraction", "seed"])
        w.writeheader()
        for r in meta_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    summary = dict(images_total=len(meta_rows),
                   splits={k: v[0] for k, v in layout.items()},
                   image_size=[IMG, IMG], channels=3,
                   mask_encoding="binary png (0/255); instance labels in /labels (16-bit)")
    with open(os.path.join(out_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Built", len(meta_rows), "images ->", out_dir)
    return out_dir, meta_rows


if __name__ == "__main__":
    build()
