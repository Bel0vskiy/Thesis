"""GMAR-L2 saliency wrapper."""

from __future__ import annotations

import numpy as np
import torch

from config import Config
from . import register
from .gmar_shared import compute_gmar_variant


@register("gmar_l2")
def compute_gmar_l2_saliency(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    image_token_positions: torch.Tensor,
    cfg: Config,
) -> dict[int, np.ndarray]:
    return compute_gmar_variant(
        model,
        tf_inputs,
        generated_ids,
        input_len,
        image_token_positions,
        cfg,
        variant="gmar_l2",
    )
