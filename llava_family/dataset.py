"""
Dataset loading for the LLaVA-family evaluation pipeline.

The public entry point is :func:`load_dataset_samples`, which returns a list
of ``{"image": PIL.Image, "caption": str, "id": str}`` dicts.  Two backends
are supported:

- **Hugging Face hub** (default) – streaming load via the ``datasets``
  library.  Column names are resolved automatically when they differ from the
  configured defaults.
- **Local directory** – reads images directly from the file system.  If a
  ``captions.csv`` file is present in the directory, captions and filenames
  are read from it; otherwise all image files are loaded in sorted order.
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

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)

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
        samples.append({
            "image": img,
            "caption": caption,
            "id": str(i),
        })

    print(f"[dataset] Loaded {len(samples)} samples.")
    return samples


def _load_local(root: str, cfg: Config) -> List[Dict]:
    import csv

    root = Path(root)
    csv_path = root / "captions.csv"

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    requested = int(cfg.num_samples)

    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            assert "filename" in fieldnames, "captions.csv must have a 'filename' column"
            cap_col = "caption" if "caption" in fieldnames else "text"
            if cap_col not in fieldnames:
                cap_col = None

            samples = []
            for row in reader:
                fname = row.get("filename", "")
                if not fname:
                    continue
                fpath = root / fname
                if not fpath.exists():
                    continue
                img = Image.open(fpath).convert("RGB")
                caption = str(row.get(cap_col, "")) if cap_col else ""
                samples.append({
                    "image": img,
                    "caption": caption,
                    "id": fname,
                })
                if len(samples) >= requested:
                    break

    else:
        files = sorted(
            p for p in root.iterdir() if p.suffix.lower() in image_extensions
        )
        samples = []
        for f in files[: requested]:
            samples.append({
                "image": Image.open(f).convert("RGB"),
                "caption": "",
                "id": f.name,
            })

    print(f"[dataset] Loaded {len(samples)} samples from {root}.")
    if len(samples) < requested:
        print(
            f"[dataset] WARNING: requested {requested} samples, but only {len(samples)} "
            f"local images/rows were found in {root}."
        )
    return samples
