"""
GMAR-L2 saliency for LLaVA models.

Thin wrapper around :func:`gmar_shared.compute_gmar_variant` that selects
the L2-norm head-weighting variant.  Head weights are proportional to the
L2 norm of the gradient of the target token logit w.r.t. each attention
head's weight matrix, then zero-floored and L1-normalised before rollout.
"""

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
