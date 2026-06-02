"""
Perturbation-based faithfulness evaluation.

For each generated token, the corresponding saliency heatmap is used to
identify the most influential image regions.  Those regions are zeroed out,
the modified image is passed through the model under teacher forcing, and the
probability drop for the target token is measured.

Three evaluation modes are provided:

* **per-token** (``evaluate_faithfulness_per_token``): each token gets its
  own masked image — one forward pass per token per mask ratio.  Produces
  token-level drop curves.
* **average** (``evaluate_faithfulness_average``): a single mask derived from
  the mean saliency across all tokens — one forward pass per mask ratio.
  Fast approximation for quick iterations.
* **random** (``evaluate_faithfulness_random``): perturbation with a
  uniformly random saliency map.  Serves as a sanity baseline.

All three modes return a dict with keys ``per_token``, ``aopc``, and
``mean_drops_by_ratio``.  AOPC is the mean probability drop averaged over
all configured mask ratios.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from model_utils import build_tf_inputs, get_token_probabilities, get_vision_tensor


# ──────────────────────────────────────────────────────────────────────────
# Masking helpers
# ──────────────────────────────────────────────────────────────────────────
def mask_pixel_values(
    pixel_values: torch.Tensor,
    saliency_map: np.ndarray,
    mask_ratio: float,
    grid_size: int = 16,
) -> torch.Tensor:
    """Zero-out the top-*mask_ratio* fraction of image regions.

    Parameters
    ----------
    pixel_values : [1, 3, H, W]
    saliency_map : (grid_size, grid_size) in [0, 1]
    mask_ratio   : fraction of grid cells to mask, e.g. 0.3
    grid_size    : must match *saliency_map* shape

    Returns
    -------
    masked : [1, 3, H, W] – a cloned tensor with masked regions set to 0.
    """
    H, W = pixel_values.shape[2], pixel_values.shape[3]

    flat = saliency_map.flatten()
    n_mask = max(1, int(mask_ratio * len(flat)))
    top_indices = np.argsort(flat)[-n_mask:]

    # grid-resolution binary mask
    grid_mask_np = np.zeros(grid_size * grid_size, dtype=np.float32)
    grid_mask_np[top_indices] = 1.0
    grid_mask_np = grid_mask_np.reshape(1, 1, grid_size, grid_size)

    # up-sample to pixel resolution (nearest-neighbour keeps hard edges)
    grid_mask_t = torch.from_numpy(grid_mask_np).to(pixel_values.device)
    pixel_mask = F.interpolate(grid_mask_t, size=(H, W), mode="nearest")
    # pixel_mask: [1, 1, H, W]  values in {0, 1}

    masked = pixel_values.clone()
    masked = masked * (1.0 - pixel_mask)
    return masked


# ──────────────────────────────────────────────────────────────────────────
# Average-saliency (fast) evaluation
# ──────────────────────────────────────────────────────────────────────────
def evaluate_faithfulness_average(
    model,
    inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    saliency_maps: dict[int, np.ndarray],
    original_probs: torch.Tensor,
    cfg: Config,
) -> dict:
    """Perturbation using the *mean* saliency map.  Very fast."""

    grid = cfg.image_token_grid
    all_maps = np.stack(list(saliency_maps.values()), axis=0)  # [N, g, g]
    avg_map = all_maps.mean(axis=0)                             # [g, g]
    # re-normalise
    lo, hi = avg_map.min(), avg_map.max()
    if hi - lo > 1e-9:
        avg_map = (avg_map - lo) / (hi - lo)

    total_len = generated_ids.shape[1]
    num_gen = total_len - input_len

    # Only evaluate tokens that belong to this saliency-map set
    var_positions = sorted(saliency_maps.keys())

    rows: list[dict] = []

    for mask_ratio in tqdm(cfg.mask_ratios, desc="  avg-perturb", leave=False):
        masked_pv = mask_pixel_values(
            get_vision_tensor(inputs), avg_map, mask_ratio, grid
        )
        tf = build_tf_inputs(inputs, generated_ids, input_len,
                             pixel_values_override=masked_pv)
        masked_probs = get_token_probabilities(model, tf, generated_ids, input_len)

        for pos in var_positions:
            tok_idx = pos - input_len
            rows.append({
                "position": pos,
                "token_id": generated_ids[0, pos].item(),
                "mask_ratio": mask_ratio,
                "original_prob": original_probs[tok_idx].item(),
                "masked_prob": masked_probs[tok_idx].item(),
                "prob_drop": original_probs[tok_idx].item() - masked_probs[tok_idx].item(),
            })

    return _aggregate(rows, cfg)


# ──────────────────────────────────────────────────────────────────────────
# Per-token (detailed) evaluation
# ──────────────────────────────────────────────────────────────────────────
def evaluate_faithfulness_per_token(
    model,
    inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    saliency_maps: dict[int, np.ndarray],
    original_probs: torch.Tensor,
    cfg: Config,
    content_mask: list[bool] | None = None,
) -> dict:
    """Per-token perturbation (one masked image per token).

    If *content_mask* is given, only tokens where the mask is ``True``
    are evaluated, which dramatically reduces runtime.
    """
    total_len = generated_ids.shape[1]
    grid = cfg.image_token_grid
    rows: list[dict] = []

    positions = list(range(input_len, total_len))
    if content_mask is not None:
        positions = [
            p for p, keep in zip(positions, content_mask) if keep
        ]

    for mask_ratio in cfg.mask_ratios:
        for pos in tqdm(
            positions,
            desc=f"  tok-perturb {mask_ratio:.0%}",
            leave=False,
        ):
            if pos not in saliency_maps:
                continue

            tok_idx = pos - input_len
            sal = saliency_maps[pos]

            masked_pv = mask_pixel_values(
                get_vision_tensor(inputs), sal, mask_ratio, grid
            )
            tf = build_tf_inputs(
                inputs, generated_ids, input_len,
                pixel_values_override=masked_pv,
            )
            masked_probs = get_token_probabilities(
                model, tf, generated_ids, input_len
            )

            rows.append({
                "position": pos,
                "token_id": generated_ids[0, pos].item(),
                "mask_ratio": mask_ratio,
                "original_prob": original_probs[tok_idx].item(),
                "masked_prob": masked_probs[tok_idx].item(),
                "prob_drop": original_probs[tok_idx].item() - masked_probs[tok_idx].item(),
            })

    return _aggregate(rows, cfg)


# ──────────────────────────────────────────────────────────────────────────
# Random baseline evaluation
# ──────────────────────────────────────────────────────────────────────────
def evaluate_faithfulness_random(
    model,
    inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    original_probs: torch.Tensor,
    cfg: Config,
    content_mask: list[bool] | None = None,
    token_positions: list[int] | None = None,
    seed: int = 42,
) -> dict:
    """Perturbation with a *random* saliency map (sanity baseline)."""
    rng = np.random.RandomState(seed)
    grid = cfg.image_token_grid
    total_len = generated_ids.shape[1]

    positions = list(range(input_len, total_len))
    if content_mask is not None:
        positions = [p for p, keep in zip(positions, content_mask) if keep]
    if token_positions is not None:
        keep = set(token_positions)
        positions = [p for p in positions if p in keep]

    rows: list[dict] = []
    for mask_ratio in tqdm(cfg.mask_ratios, desc="  rand-perturb", leave=False):
        rand_map = rng.rand(grid, grid).astype(np.float32)
        masked_pv = mask_pixel_values(
            get_vision_tensor(inputs), rand_map, mask_ratio, grid,
        )
        tf = build_tf_inputs(inputs, generated_ids, input_len,
                             pixel_values_override=masked_pv)
        masked_probs = get_token_probabilities(model, tf, generated_ids, input_len)

        for pos in positions:
            tok_idx = pos - input_len
            rows.append({
                "position": pos,
                "token_id": generated_ids[0, pos].item(),
                "mask_ratio": mask_ratio,
                "original_prob": original_probs[tok_idx].item(),
                "masked_prob": masked_probs[tok_idx].item(),
                "prob_drop": original_probs[tok_idx].item() - masked_probs[tok_idx].item(),
            })

    return _aggregate(rows, cfg)


# ──────────────────────────────────────────────────────────────────────────
# Shared aggregation
# ──────────────────────────────────────────────────────────────────────────
def _aggregate(rows: list[dict], cfg: Config) -> dict:
    if not rows:
        return {"per_token": [], "aopc": 0.0, "mean_drops_by_ratio": {}}

    mean_by_ratio: dict[float, float] = {}
    for mr in cfg.mask_ratios:
        drops = [r["prob_drop"] for r in rows if r["mask_ratio"] == mr]
        if drops:
            mean_by_ratio[mr] = float(np.mean(drops))

    aopc = float(np.mean(list(mean_by_ratio.values()))) if mean_by_ratio else 0.0

    return {
        "per_token": rows,
        "aopc": aopc,
        "mean_drops_by_ratio": mean_by_ratio,
    }
