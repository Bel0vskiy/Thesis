"""
Attention Rollout saliency extraction for LLaVA models.

Implements Attention Rollout (Abnar & Zuidema, 2020) which propagates
attention through transformer layers by recursively multiplying
attention matrices, accounting for residual connections:

    R_0 = I
    R_l = 0.5 * A_l · R_{l-1}  +  0.5 * I      (residual blend)

For each generated token the rollout row is extracted at the image-token
columns and reshaped to the spatial grid.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from . import register


def _first_real_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def _prepare_forward_kwargs(model, tf_inputs: dict) -> dict:
    kwargs = dict(tf_inputs)
    is_official_llava = model.__class__.__module__.startswith("llava.")

    if is_official_llava and "pixel_values" in kwargs and "images" not in kwargs:
        kwargs["images"] = kwargs.pop("pixel_values")

    if "images" in kwargs and torch.is_tensor(kwargs["images"]):
        img = kwargs["images"]
        model_dev = _first_real_device(model)
        model_dtype = None
        for p in model.parameters():
            if p.device.type != "meta":
                model_dtype = p.dtype
                break
        if model_dtype is None:
            model_dtype = torch.float16

        img = img.to(device=model_dev)
        if torch.is_floating_point(img):
            img = img.to(dtype=model_dtype)
        kwargs["images"] = img

    return kwargs


def _sanitize_image_positions(
    image_token_positions: torch.Tensor,
    seq_len: int,
    grid: int,
) -> torch.Tensor:
    target = int(grid * grid)
    if seq_len <= 0:
        return torch.empty(0, dtype=torch.long)

    img_pos = image_token_positions.detach().cpu().long().flatten()
    valid = img_pos[(img_pos >= 0) & (img_pos < int(seq_len))]
    if len(valid) == target:
        return valid

    # Some backends include a leading CLS-like visual token.
    if len(valid) == target + 1:
        return valid[1:]

    # If we have extra candidates, prefer the last contiguous target tokens.
    if len(valid) > target + 1:
        return valid[-target:]

    return torch.empty(0, dtype=torch.long)


def _postprocess_attention_maps(
    raw_maps: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for pos, raw in raw_maps.items():
        sal = raw.astype(np.float32, copy=True)
        lo = sal.min()
        hi = np.percentile(sal, 99)
        if hi - lo > 1e-9:
            sal = np.clip((sal - lo) / (hi - lo), 0.0, 1.0)
        else:
            hi = sal.max()
            if hi - lo > 1e-9:
                sal = (sal - lo) / (hi - lo)
        out[pos] = sal.astype(np.float32)
    return out


def _attention_rollout(
    attentions: tuple[torch.Tensor, ...],
    layer_ids: list[int],
    block_key_from: int | None = None,
) -> torch.Tensor:
    """Compute attention rollout over selected layers.

    This matches MedGemma's rollout formulation. Optional generated-token
    key blocking is retained as a LLaVA-specific diagnostic control.
    """
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device

    rollout = torch.eye(seq_len, device=device, dtype=torch.float32)
    I = torch.eye(seq_len, device=device, dtype=torch.float32)

    for li in layer_ids:
        attn = attentions[li][0].float()   # [H, S, S]
        a = attn.mean(dim=0)               # [S, S] — uniform head average

        if block_key_from is not None and 0 <= int(block_key_from) < seq_len:
            a[:, int(block_key_from):] = 0.0

        # Residual blend + row-renormalise
        a = 0.5 * a + 0.5 * I
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)

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
    """Return ``{abs_position: grid×grid numpy saliency map}``."""

    total_len = generated_ids.shape[1]
    grid = cfg.image_token_grid

    # ── forward with attention output ────────────────────────────────────
    fwd_kwargs = _prepare_forward_kwargs(model, tf_inputs)
    with torch.no_grad():
        outputs = model(**fwd_kwargs, output_attentions=True)

    attentions = outputs.attentions   # tuple of [1, H, S, S]
    num_layers = len(attentions)
    seq_len = int(attentions[0].shape[-1])

    # ── choose layers ─────────────────────────────────────────────────────
    if cfg.attention_layer_strategy == "all":
        layer_ids = list(range(num_layers))
    elif cfg.attention_layer_strategy.startswith("last"):
        n = int(cfg.attention_layer_strategy[4:])
        pool = [i for i in cfg.global_attn_layers if i < num_layers]
        layer_ids = pool[-n:] if n <= len(pool) else pool
    else:  # "global" (default)
        layer_ids = [i for i in cfg.global_attn_layers if i < num_layers]

    if not layer_ids:
        layer_ids = list(range(num_layers))

    # ── image token positions ─────────────────────────────────────────────
    img_pos = _sanitize_image_positions(image_token_positions, seq_len, grid)
    if len(img_pos) != grid * grid:
        print(
            f"[attention] invalid image token mapping: expected {grid * grid}, "
            f"got {len(img_pos)} (seq_len={seq_len}). Skipping sample."
        )
        return {}

    # LLaVA-Med expands the -200 placeholder to 576 patch tokens during
    # forward, so the rollout matrix has seq_len = total_len + expansion_offset.
    expansion_offset = seq_len - total_len

    # Literature-faithful default keeps generated-token inheritance enabled.
    # Blocking is available only as an explicit diagnostic override.
    block_generated = bool(getattr(cfg, "rollout_block_generated_tokens", False))
    block_key_from = input_len + expansion_offset if block_generated else None

    # ── attention rollout ────────────────────────────────────────────────
    rollout = _attention_rollout(attentions, layer_ids, block_key_from=block_key_from)

    img_pos_t = img_pos.to(rollout.device)
    raw_maps: dict[int, np.ndarray] = {}

    for pos in tqdm(range(input_len, total_len), desc="  attn saliency", leave=False):
        rollout_idx = pos + expansion_offset
        if rollout_idx < 0 or rollout_idx >= seq_len:
            continue
        row = rollout[rollout_idx, img_pos_t].cpu().float().numpy()
        raw_maps[pos] = row.reshape(grid, grid)

    del rollout, attentions, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _postprocess_attention_maps(raw_maps)
