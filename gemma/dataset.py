"""
Dataset loading utilities.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from PIL import Image
from tqdm import tqdm

from config import Config

def load_dataset_samples(cfg: Config) -> List[Dict]:
    """Return a list of ``{image, caption, id}`` dicts.

    Tries HuggingFace first; falls back to a local directory.
    """
    path = cfg.dataset_name
    if os.path.isdir(path):
        return _load_local(path, cfg)
    return _load_huggingface(cfg)

def _load_huggingface(cfg: Config) -> List[Dict]:
    from datasets import load_dataset

    print(f"[dataset] Loading '{cfg.dataset_name}' split='{cfg.dataset_split}' "
          f"(streaming, {cfg.num_samples} samples) …")

    load_kwargs = dict(split=cfg.dataset_split, streaming=True)
    if getattr(cfg, "dataset_config", ""):
        ds = load_dataset(cfg.dataset_name, cfg.dataset_config, **load_kwargs)
    else:
        ds = load_dataset(cfg.dataset_name, **load_kwargs)

    # Peek at the first row to resolve column names
    first = next(iter(ds))
    column_names = list(first.keys())

    img_col = cfg.image_column
    if img_col not in column_names:
        for alt in ("image", "img", "pixel_values"):
            if alt in column_names:
                img_col = alt
                break
        else:
            raise KeyError(
                f"Cannot find an image column in {column_names}. "
                f"Set cfg.image_column explicitly."
            )

    cap_col = cfg.caption_column
    if cap_col not in column_names:
        for alt in ("caption", "text", "description", "label"):
            if alt in column_names:
                cap_col = alt
                break
        else:
            cap_col = None
            print("[dataset] WARNING: no caption column found – "
                  "reference captions will be empty.")

    samples = []
    for i, row in enumerate(tqdm(ds, desc="Loading samples", total=cfg.num_samples)):
        if i >= cfg.num_samples:
            break
        img = row[img_col]
        if isinstance(img, str):
            img = Image.open(img)
        img = img.convert("RGB")
        caption = row[cap_col] if cap_col else ""
        # Some datasets (e.g. COCO) store multiple captions as a list — take the first
        if isinstance(caption, (list, tuple)):
            caption = caption[0] if caption else ""
        samples.append({
            "image": img,
            "caption": caption,
            "id": str(i),
        })

    print(f"[dataset] Loaded {len(samples)} samples.")
    return samples


def _load_local(root: str, cfg: Config) -> List[Dict]:
    import pandas as pd

    root = Path(root)
    csv_path = root / "captions.csv"

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        assert "filename" in df.columns, "captions.csv must have a 'filename' column"
        cap_col = "caption" if "caption" in df.columns else "text"
        if cap_col not in df.columns:
            cap_col = None

        samples = []
        for _, row in df.iterrows():
            fpath = root / row["filename"]
            if not fpath.exists():
                continue
            img = Image.open(fpath).convert("RGB")
            caption = str(row[cap_col]) if cap_col else ""
            samples.append({
                "image": img,
                "caption": caption,
                "id": row["filename"],
            })
            if len(samples) >= cfg.num_samples:
                break
    else:
        files = sorted(
            p for p in root.iterdir() if p.suffix.lower() in image_extensions
        )
        samples = []
        for f in files[: cfg.num_samples]:
            samples.append({
                "image": Image.open(f).convert("RGB"),
                "caption": "",
                "id": f.name,
            })

    print(f"[dataset] Loaded {len(samples)} samples from {root}.")
    return samples
