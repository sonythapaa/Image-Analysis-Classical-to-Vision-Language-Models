"""Load the nuclei dataset, convert to grayscale,
resize to a common size, and produce EDA figures (sample grid + intensity
histogram).

The dataset ships at 256x256 already, so "resize to 256x256" is a no-op for
this particular set — we still apply it explicitly so the pipeline works
unchanged if someone points it at a differently-sized modality.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TARGET_SIZE = (256, 256)


@dataclass
class DatasetPaths:
    root: Path

    @property
    def metadata_csv(self) -> Path:
        return self.root / "metadata.csv"

    def images_dir(self, split: str) -> Path:
        return self.root / split / "images"

    def masks_dir(self, split: str) -> Path:
        return self.root / split / "masks"

    def labels_dir(self, split: str) -> Path:
        return self.root / split / "labels"


def load_metadata(paths: DatasetPaths) -> pd.DataFrame:
    """Load the ground-truth metadata table shipped with the dataset."""
    return pd.read_csv(paths.metadata_csv)


def load_and_preprocess(image_path: Path, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Load one image, convert to grayscale, resize to `size`.

    Returns a float32 array in [0, 1], shape (H, W).
    """
    img = Image.open(image_path).convert("L")  # grayscale
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def preprocess_split(paths: DatasetPaths, split: str, size: tuple[int, int] = TARGET_SIZE) -> dict[str, np.ndarray]:
    """Preprocess every image in a split. Returns {image_id: array}."""
    out: dict[str, np.ndarray] = {}
    for p in sorted(paths.images_dir(split).glob("*.png")):
        out[p.stem] = load_and_preprocess(p, size)
    return out


def intensity_histogram_data(images: dict[str, np.ndarray], bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Pooled pixel-intensity histogram across a dict of preprocessed images."""
    all_pixels = np.concatenate([im.ravel() for im in images.values()])
    counts, edges = np.histogram(all_pixels, bins=bins, range=(0.0, 1.0))
    return counts, edges


def eda_summary_table(meta: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    df = meta if split is None else meta[meta["split"] == split]
    return (
        df.groupby("density")[["n_objects", "mean_intensity", "area_fraction"]]
        .agg(["mean", "std", "count"])
        .round(3)
    )



def ensure_noise_corruption(paths: DatasetPaths, base_ids: tuple[str, ...] = ("test_000", "test_004"),
                              sigma: float = 0.15, seed: int = 42) -> None:
    """Generate a synthetic additive-Gaussian-noise corrupted variant for
    each of `base_ids`, saved alongside the dataset's own pre-built
    blur/lowcontrast variants in test_corrupted/images/ following the same
    `<base_id>_<variant>.png` naming convention, so robustness_trace() picks
    them up automatically with no other code changes needed.

    This is the third corruption type the assignment brief explicitly
    suggests ("heavy blur, low contrast, or added noise") that the
    dataset's own pre-built test_corrupted/ folder doesn't include —
    generated here in code, seeded, so it's reproducible rather than a
    one-off manual edit.
    """
    import numpy as np
    from PIL import Image

    corrupted_dir = paths.root / "test_corrupted" / "images"
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for base_id in base_ids:
        out_path = corrupted_dir / f"{base_id}_noise.png"
        if out_path.exists():
            continue
        clean_path = paths.images_dir("test") / f"{base_id}.png"
        arr = np.asarray(Image.open(clean_path).convert("L"), dtype=np.float32) / 255.0
        noisy = arr + rng.normal(loc=0.0, scale=sigma, size=arr.shape)
        noisy = np.clip(noisy, 0.0, 1.0)
        Image.fromarray((noisy * 255).astype("uint8")).save(out_path)
