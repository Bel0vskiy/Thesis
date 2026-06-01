"""
Shared GMAR computation for MedGemma.

Computes GMAR-L1 and GMAR-L2 saliency maps in one heavy pass
(single forward + per-token backward loop), and caches the latest
sample result so requesting gmar_l1 then gmar_l2 does not duplicate
work.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..config import Config

_LAST_KEY: tuple | None = None
_LAST_VALUE: dict[str, dict[int, np.ndarray]] | None = None


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
        return weights / total
    return torch.ones(num_heads, device=weights.device) / num_heads


def _attention_rollout(
    attentions: list[torch.Tensor],
    head_weights: list[torch.Tensor],
    layer_ids: list[int],
) -> torch.Tensor:
    seq_len = attentions[0].shape[-1]
    device = attentions[0].device
    rollout = torch.eye(seq_len, device=device, dtype=torch.float32)

    for layer_index in layer_ids:
        attention = attentions[layer_index][0].float()
        weights = head_weights[layer_index]
        rolled = (weights[:, None, None] * attention).sum(dim=0)
        rolled = 0.5 * rolled + 0.5 * torch.eye(seq_len, device=device, dtype=torch.float32)
        rolled = rolled / (rolled.sum(dim=-1, keepdim=True) + 1e-9)
        rollout = rolled @ rollout

    return rollout


def _postprocess(raw_maps: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for pos, raw in raw_maps.items():
        # Keep raw token saliency (no cross-token baseline subtraction).
        sal = np.clip(raw.copy(), 0.0, None)
        # Match attention-rollout sink suppression for MedGemma.
        sal[12, 0] = 0.0

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

    outputs = model(**tf_inputs, output_attentions=True)
    logits = outputs.logits
    attentions = outputs.attentions
    attn_detached = [attention.detach() for attention in attentions]

    num_layers = len(attn_detached)
    num_heads = attn_detached[0].shape[1]
    layer_ids = _select_layers(cfg, num_layers)

    img_pos_t = img_pos.to(attn_detached[0].device)
    gen_positions = list(range(input_len, total_len))

    raw_l1: dict[int, np.ndarray] = {}
    raw_l2: dict[int, np.ndarray] = {}

    for index, pos in enumerate(tqdm(gen_positions, desc="  gmar saliency", leave=False)):
        target_id = generated_ids[0, pos].item()
        score = logits[0, pos - 1, target_id]

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

        rollout_l1 = _attention_rollout(attn_detached, weights_l1, layer_ids)
        rollout_l2 = _attention_rollout(attn_detached, weights_l2, layer_ids)

        raw_l1[pos] = rollout_l1[pos, img_pos_t].cpu().float().numpy().reshape(grid, grid)
        raw_l2[pos] = rollout_l2[pos, img_pos_t].cpu().float().numpy().reshape(grid, grid)

    del attentions, attn_detached, outputs, logits
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "gmar_l1": _postprocess(raw_l1),
        "gmar_l2": _postprocess(raw_l2),
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
