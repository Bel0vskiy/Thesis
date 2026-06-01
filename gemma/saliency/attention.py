"""
Attention Rollout saliency extraction.

Implements Attention Rollout (Abnar & Zuidema, 2020) which propagates
attention through transformer layers by recursively multiplying
attention matrices, accounting for residual connections:

    R_0 = I
    R_l = 0.5 * A_l · R_{l-1}  +  0.5 * I      (residual blend)

For each generated token the rollout row is extracted at the image-token
columns and reshaped to the spatial grid.

A single teacher-forcing forward pass with ``output_attentions=True`` is
used, so this method is fast and memory-lean.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from . import register


def _attention_rollout(
    attentions: tuple[torch.Tensor, ...],
    layer_ids: list[int],
) -> torch.Tensor:
    """Compute attention rollout over the selected layers.

    Parameters
    ----------
    attentions : tuple of [1, H, S, S]
    layer_ids : which layers to include in the rollout

    Returns
    -------
    rollout : [S, S] – the accumulated attention rollout matrix.
    """
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device

    rollout = torch.eye(seq_len, device=device, dtype=torch.float32)

    for li in layer_ids:
        attn = attentions[li][0].float()            # [H, S, S]

        # Uniform average across heads
        a = attn.mean(dim=0)                        # [S, S]

        # Add residual connection (identity) and renormalise rows
        a = 0.5 * a + 0.5 * torch.eye(seq_len, device=device, dtype=torch.float32)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)

        # Accumulate rollout
        rollout = a @ rollout

    return rollout


@register("attention")
def compute_attention_saliency(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    image_token_positions: torch.Tensor,
    cfg: Config,
) -> dict[int, np.ndarray]:
    """Return ``{abs_position: 16×16 numpy saliency map}``."""

    total_len = generated_ids.shape[1]
    grid = cfg.image_token_grid                     # 16

    # ── forward with attention output ────────────────────────────────────
    with torch.no_grad():
        outputs = model(**tf_inputs, output_attentions=True)

    attentions = outputs.attentions                 # tuple of [1, H, S, S]
    num_layers = len(attentions)

    # ── choose which layers to aggregate over ────────────────────────────
    if cfg.attention_layer_strategy == "global":
        layer_ids = [i for i in cfg.global_attn_layers if i < num_layers]
    elif cfg.attention_layer_strategy == "all":
        layer_ids = list(range(num_layers))
    elif cfg.attention_layer_strategy.startswith("last"):
        n = int(cfg.attention_layer_strategy[4:])
        pool = [i for i in cfg.global_attn_layers if i < num_layers]
        layer_ids = pool[-n:] if n <= len(pool) else pool
    else:
        layer_ids = [i for i in cfg.global_attn_layers if i < num_layers]

    if not layer_ids:
        layer_ids = list(range(num_layers))         # safety fallback

    img_pos = image_token_positions.cpu()
    n_img = len(img_pos)
    assert n_img == grid * grid, (
        f"Expected {grid * grid} image tokens, got {n_img}"
    )

    # ── attention rollout ────────────────────────────────────────────────
    rollout = _attention_rollout(attentions, layer_ids)  # [S, S]

    # Free the (potentially large) attention tensors
    del attentions, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── extract per-token saliency from rollout ─────────────────────────
    img_pos_t = img_pos.to(rollout.device)
    saliency_maps: dict[int, np.ndarray] = {}

    for pos in tqdm(
        range(input_len, total_len), desc="  attn saliency", leave=False
    ):
        row = rollout[pos, img_pos_t].cpu().float().numpy()  # [n_img]
        sal = row.reshape(grid, grid)

        # Zero out known attention-sink position
        sal[12, 0] = 0.0

        # Percentile-clipped normalisation — prevents attention-sink
        # positions from dominating the colour scale.
        lo = sal.min()
        hi = np.percentile(sal, 99)
        if hi - lo > 1e-9:
            sal = np.clip((sal - lo) / (hi - lo), 0.0, 1.0)
        else:
            hi = sal.max()
            if hi - lo > 1e-9:
                sal = (sal - lo) / (hi - lo)
            # else: leave as-is (all values essentially equal)

        saliency_maps[pos] = sal.astype(np.float32)

    return saliency_maps
