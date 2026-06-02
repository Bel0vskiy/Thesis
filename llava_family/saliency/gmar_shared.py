"""
Shared GMAR computation for LLaVA-family models.

Computes GMAR-L1 and GMAR-L2 saliency maps in a single pass
(one forward pass followed by a per-token backward loop) and caches the
result for the current sample so that requesting gmar_l1 and gmar_l2
back-to-back does not repeat the expensive computation.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from config import Config

_LAST_KEY: tuple | None = None
_LAST_VALUE: dict[str, dict[int, np.ndarray]] | None = None


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


def _cache_key(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    image_token_positions: torch.Tensor,
    cfg: Config,
) -> tuple:
    return (
        id(model),
        id(tf_inputs),
        id(generated_ids),
        input_len,
        tuple(image_token_positions.detach().cpu().tolist()),
        cfg.image_token_grid,
        cfg.attention_layer_strategy,
        tuple(cfg.global_attn_layers),
    )


def _select_layers(cfg: Config, num_layers: int) -> list[int]:
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
        layer_ids = list(range(num_layers))
    return layer_ids


def _normalize_head_weights(weights: torch.Tensor, num_heads: int) -> torch.Tensor:
    weights = weights.clamp(min=0)
    total = weights.sum()
    if total > 1e-9:
        # Softmax with temperature = mean weight, so deviations are amplified
        # relative to the uniform baseline.  This makes gradient-weighted heads
        # more distinct from plain rollout while preserving rank order.
        mean_w = total / num_heads
        return torch.softmax(weights / mean_w, dim=0)
    return torch.ones(num_heads, device=weights.device) / num_heads


def _attention_rollout(
    attentions: list[torch.Tensor],
    head_weights: list[torch.Tensor],
    layer_ids: list[int],
    block_key_from: int | None = None,
) -> torch.Tensor:
    """Gradient-weighted attention rollout (GMAR).

    Head weights are derived from gradients, which naturally down-weight sink
    heads (heads whose patterns are query-independent contribute near-zero
    gradient).  No additional sink filtering is applied here — doing so would
    double-penalise sink heads and corrupt the gradient signal.

    Optionally, generated-token key columns can be blocked (``block_key_from``)
    for diagnostic runs that suppress generated-token attribution chains.
    """
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device
    rollout = torch.eye(seq_len, device=device, dtype=torch.float32)
    I = torch.eye(seq_len, device=device, dtype=torch.float32)

    for layer_index in layer_ids:
        attention = attentions[layer_index][0].float()   # [H, S, S]
        weights = head_weights[layer_index]              # [H]

        rolled = (weights[:, None, None] * attention).sum(dim=0)

        if block_key_from is not None and 0 <= int(block_key_from) < seq_len:
            rolled[:, int(block_key_from):] = 0.0

        rolled = 0.5 * rolled + 0.5 * I
        rolled = rolled / (rolled.sum(dim=-1, keepdim=True) + 1e-9)
        rollout = rolled @ rollout

    return rollout


def _postprocess(
    raw_maps: dict[int, np.ndarray],
    cfg: Config,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for pos, raw in raw_maps.items():
        sal = np.clip(raw.copy(), 0.0, None)
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


def _sanitize_image_positions(
    image_token_positions: torch.Tensor,
    seq_len: int,
    grid: int,
    input_len: int,
) -> torch.Tensor:
    """Return exactly grid*grid valid image-token positions.

    Official LLaVA-Med variants may expose a single placeholder in text ids
    while attention tensors are over an expanded multimodal sequence.
    This utility clamps/remaps to a safe contiguous block when needed.
    """
    target = int(grid * grid)
    if seq_len <= 0:
        return torch.empty(0, dtype=torch.long)

    img_pos = image_token_positions.detach().cpu().long().flatten()
    valid = img_pos[(img_pos >= 0) & (img_pos < int(seq_len))]

    if len(valid) == target:
        return valid

    if len(valid) == target + 1:
        return valid[1:]

    if len(valid) > target + 1:
        return valid[-target:]

    return torch.empty(0, dtype=torch.long)


def _compute_pair(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    image_token_positions: torch.Tensor,
    cfg: Config,
) -> dict[str, dict[int, np.ndarray]]:
    total_len = generated_ids.shape[1]
    grid = cfg.image_token_grid

    img_pos = image_token_positions.cpu()
    n_img = len(img_pos)
    assert n_img == grid * grid, f"Expected {grid * grid} image tokens, got {n_img}"

    outputs = model(**_prepare_forward_kwargs(model, tf_inputs), output_attentions=True)
    logits = outputs.logits
    attentions = outputs.attentions
    attn_detached = [attention.detach() for attention in attentions]
    seq_len = int(attn_detached[0].shape[-1])
    logit_len = int(logits.shape[1])
    vocab_size = int(logits.shape[-1])

    num_layers = len(attn_detached)
    num_heads = attn_detached[0].shape[1]
    layer_ids = _select_layers(cfg, num_layers)

    # Ensure image positions are always valid for the current attention length.
    img_pos = _sanitize_image_positions(image_token_positions, seq_len, grid, input_len)
    n_img = len(img_pos)
    if n_img != grid * grid:
        print(
            f"[gmar] invalid image token mapping: expected {grid * grid}, "
            f"got {n_img} (seq_len={seq_len}, input_len={input_len}). Skipping sample."
        )
        return {
            "gmar_l1": {},
            "gmar_l2": {},
        }

    img_pos_t = img_pos.to(attn_detached[0].device)

    # LLaVA-Med expands -200 → 576 patch tokens; logits and rollout are
    # over the expanded sequence. Shift unexpanded positions by the offset.
    expansion_offset = seq_len - total_len   # 575 for LLaVA-Med 24x24, 0 otherwise

    # Keep only generated-token positions safe for both logit and rollout indexing.
    # This matches MedGemma indexing semantics while accounting for LLaVA
    # placeholder expansion.
    gen_positions = [
        pos
        for pos in range(input_len, total_len)
        if 0 <= (pos - 1 + expansion_offset) < logit_len
        and 0 <= (pos + expansion_offset) < seq_len
    ]

    if not gen_positions:
        return {
            "gmar_l1": {},
            "gmar_l2": {},
        }

    block_generated = bool(getattr(cfg, "rollout_block_generated_tokens", False))
    block_key_from = input_len + expansion_offset if block_generated else None

    raw_l1: dict[int, np.ndarray] = {}
    raw_l2: dict[int, np.ndarray] = {}

    for index, pos in enumerate(tqdm(gen_positions, desc="  gmar saliency", leave=False)):
        target_id = generated_ids[0, pos].item()
        if not (0 <= int(target_id) < vocab_size):
            continue
        score = logits[0, pos - 1 + expansion_offset, target_id]

        grads = torch.autograd.grad(
            score,
            attentions,
            retain_graph=index < len(gen_positions) - 1,
            create_graph=False,
        )

        weights_l1: list[torch.Tensor] = []
        weights_l2: list[torch.Tensor] = []
        for grad in grads:
            g = grad[0]
            w1 = _normalize_head_weights(g.abs().mean(dim=(-2, -1)), num_heads)
            w2 = _normalize_head_weights(g.square().mean(dim=(-2, -1)).sqrt(), num_heads)
            weights_l1.append(w1.detach())
            weights_l2.append(w2.detach())

        rollout_l1 = _attention_rollout(
            attn_detached,
            weights_l1,
            layer_ids,
            block_key_from=block_key_from,
        )
        rollout_l2 = _attention_rollout(
            attn_detached,
            weights_l2,
            layer_ids,
            block_key_from=block_key_from,
        )

        rollout_idx = pos + expansion_offset
        raw_l1[pos] = rollout_l1[rollout_idx, img_pos_t].cpu().float().numpy().reshape(grid, grid)
        raw_l2[pos] = rollout_l2[rollout_idx, img_pos_t].cpu().float().numpy().reshape(grid, grid)

    if not raw_l1 or not raw_l2:
        return {
            "gmar_l1": {},
            "gmar_l2": {},
        }

    del attentions, attn_detached, outputs, logits
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "gmar_l1": _postprocess(raw_l1, cfg),
        "gmar_l2": _postprocess(raw_l2, cfg),
    }


def compute_gmar_variant(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    image_token_positions: torch.Tensor,
    cfg: Config,
    variant: str,
) -> dict[int, np.ndarray]:
    if variant not in {"gmar_l1", "gmar_l2"}:
        raise ValueError(f"Unknown GMAR variant: {variant}")

    global _LAST_KEY, _LAST_VALUE
    key = _cache_key(model, tf_inputs, generated_ids, input_len, image_token_positions, cfg)
    if _LAST_KEY != key or _LAST_VALUE is None:
        _LAST_VALUE = _compute_pair(
            model, tf_inputs, generated_ids, input_len, image_token_positions, cfg
        )
        _LAST_KEY = key

    return {k: v.copy() for k, v in _LAST_VALUE[variant].items()}
