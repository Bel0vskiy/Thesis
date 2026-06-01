"""
Visualization utilities for saliency maps and perturbation results.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from .config import Config


_METHOD_LABELS = {
    "attention": "Attention",
    "gradcam": "GradCAM",
    "gmar_l1": "GMAR L1",
    "gmar_l2": "GMAR L2",
    "random": "Random",
}


def _method_label(name: str) -> str:
    return _METHOD_LABELS.get(name, name.replace("_", " ").title())


def _normalize_ratio_dict(drops: dict) -> dict[float, float]:
    """Return a ratio->value dict with float keys/values.

    This guards against mixed key types caused by JSON round-trips
    (e.g. keys as strings on disk and floats in-memory).
    """
    out: dict[float, float] = {}
    if not isinstance(drops, dict):
        return out

    for k, v in drops.items():
        try:
            ratio = float(k)
            value = float(v)
        except (TypeError, ValueError):
            continue
        out[ratio] = value
    return out


# ──────────────────────────────────────────────────────────────────────────
# Heatmap overlay
# ──────────────────────────────────────────────────────────────────────────
def overlay_heatmap(
    image: Image.Image,
    saliency: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> np.ndarray:
    """Return an RGB numpy array with the saliency heatmap blended onto
    *image*.

    *saliency* is a 2-D array (any resolution); it is up-scaled to *image*
    size automatically.
    """
    img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]

    # up-scale saliency to image resolution
    sal_pil = Image.fromarray((saliency * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR
    )
    sal_np = np.array(sal_pil).astype(np.float32) / 255.0

    cm = plt.get_cmap(cmap)
    heatmap = cm(sal_np)[..., :3]                   # drop alpha channel

    blended = (1 - alpha) * img_np + alpha * heatmap
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
# Per-sample visualisation
# ──────────────────────────────────────────────────────────────────────────
def save_token_saliency_grid(
    image: Image.Image,
    saliency_maps: dict[int, np.ndarray],
    token_strings: dict[int, str],
    method_name: str,
    save_path: str,
    max_tokens: int = 12,
    content_positions: list[int] | None = None,
):
    """Save a figure showing the original image + overlaid saliency for
    the first *max_tokens* content tokens.

    If *content_positions* is given, only those positions are shown
    (skipping stopwords, punctuation, etc.).
    """

    if content_positions is not None:
        # Only show content tokens that have a saliency map
        positions = [p for p in content_positions if p in saliency_maps][:max_tokens]
    else:
        positions = sorted(saliency_maps.keys())[:max_tokens]
    n = len(positions)
    if n == 0:
        return

    cols = min(n + 1, 7)
    rows = 1 + (n) // (cols)       # first cell = original image
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)

    # original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original", fontsize=8)
    axes[0, 0].axis("off")

    for idx, pos in enumerate(positions):
        r, c = divmod(idx + 1, cols)
        ax = axes[r, c]
        blended = overlay_heatmap(image, saliency_maps[pos])
        ax.imshow(blended)
        tok_str = token_strings.get(pos, f"[{pos}]")
        ax.set_title(f'"{tok_str}"', fontsize=7)
        ax.axis("off")

    # hide empty axes
    for r in range(rows):
        for c in range(cols):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    fig.suptitle(f"Saliency - {_method_label(method_name)}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Comparison figure (Attention vs Grad-CAM side by side)
# ──────────────────────────────────────────────────────────────────────────
def save_comparison_figure(
    image: Image.Image,
    all_saliency: dict[str, dict[int, np.ndarray]],
    token_strings: dict[int, str],
    save_path: str,
    max_tokens: int = 8,
    content_positions: list[int] | None = None,
):
    """Side-by-side comparison of multiple saliency methods for the same
    tokens."""

    methods = list(all_saliency.keys())
    # pick positions common to all methods
    common = set.intersection(*(set(maps.keys()) for maps in all_saliency.values()))
    if content_positions is not None:
        positions = [p for p in content_positions if p in common][:max_tokens]
    else:
        positions = sorted(common)[:max_tokens]
    n = len(positions)
    if n == 0:
        return

    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_methods + 1, n,
        figsize=(2.5 * n, 2.8 * (n_methods + 1)),
    )
    if n == 1:
        axes = axes[:, np.newaxis]

    # Row 0: original images (repeated for alignment)
    for j, pos in enumerate(positions):
        axes[0, j].imshow(image)
        tok = token_strings.get(pos, f"[{pos}]")
        axes[0, j].set_title(f'"{tok}"', fontsize=7)
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=8)

    for mi, method in enumerate(methods):
        for j, pos in enumerate(positions):
            ax = axes[mi + 1, j]
            if pos in all_saliency[method]:
                blended = overlay_heatmap(image, all_saliency[method][pos])
                ax.imshow(blended)
            ax.axis("off")
        axes[mi + 1, 0].set_ylabel(_method_label(method), fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Perturbation curve
# ──────────────────────────────────────────────────────────────────────────
def save_perturbation_curve(
    results_by_method: dict[str, dict],
    save_path: str,
    title: str = "Perturbation Curve",
):
    """Plot mean probability-drop vs. mask ratio for each method.

    *results_by_method* maps ``method_name`` → evaluation result dict
    (as returned by ``evaluate_faithfulness_*``).  Each dict must contain
    ``mean_drops_by_ratio``.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, (method, res) in enumerate(results_by_method.items()):
        drops = _normalize_ratio_dict(res.get("mean_drops_by_ratio", {}))
        if not drops:
            continue
        ratios = sorted(drops.keys())
        values = [drops[r] for r in ratios]
        ax.plot(
            [r * 100 for r in ratios],
            values,
            marker=markers[i % len(markers)],
            label=f"{_method_label(method)}  (AOPC={res.get('aopc', 0):.4f})",
        )

    ax.set_xlabel("Mask ratio (%)")
    ax.set_ylabel("Mean probability drop")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Aggregate summary across all samples
# ──────────────────────────────────────────────────────────────────────────
def save_aggregate_curves(
    all_results: dict[str, list[dict]],
    save_path: str,
):
    """
    Parameters
    ----------
    all_results : {method: [sample_result_dict, …]}
        Each ``sample_result_dict`` has ``mean_drops_by_ratio``.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, (method, sample_results) in enumerate(all_results.items()):
        if not sample_results:
            continue

        norm_sample_results = []
        for sr in sample_results:
            norm_drops = _normalize_ratio_dict(sr.get("mean_drops_by_ratio", {}))
            if not norm_drops:
                continue
            norm_sample_results.append({
                "drops": norm_drops,
                "aopc": float(sr.get("aopc", 0.0)),
            })

        if not norm_sample_results:
            continue

        # Collect all ratios present
        all_ratios = set()
        for sr in norm_sample_results:
            all_ratios.update(sr["drops"].keys())

        ratios = sorted(all_ratios)
        means = []
        stds = []
        for r in ratios:
            vals = [
                sr["drops"][r]
                for sr in norm_sample_results
                if r in sr["drops"]
            ]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        means = np.array(means)
        stds = np.array(stds)
        xs = [r * 100 for r in ratios]
        aopc_vals = [sr["aopc"] for sr in norm_sample_results]
        mean_aopc = np.mean(aopc_vals)

        ax.plot(
            xs, means,
            marker=markers[i % len(markers)],
            label=f"{_method_label(method)}  (AOPC={mean_aopc:.4f})",
        )
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            alpha=0.15,
        )

    ax.set_xlabel("Mask ratio (%)")
    ax.set_ylabel("Mean probability drop")
    ax.set_title("Aggregate perturbation curves (mean +/- 1 std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
