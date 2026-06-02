"""
GMAR-L1 saliency for Gemma-family models.

Thin wrapper around :func:`gmar_shared.compute_gmar_variant` that selects
the L1-norm head-weighting variant.  Head weights are proportional to the
L1 norm of the gradient of the target token logit w.r.t. each attention
head's weight matrix, then zero-floored and L1-normalised before rollout.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
from . import register
from .gmar_shared import compute_gmar_variant


@register("gmar_l1")
def compute_gmar_l1_saliency(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    image_token_positions: torch.Tensor,
    cfg: Config,
) -> dict[int, np.ndarray]:
    """Return ``{abs_position: grid×grid saliency map}`` using GMAR-L1.

    Head importance weights are computed as the mean absolute gradient of
    each attention head's matrix w.r.t. the target token's logit.  The
    weighted attention rollout is then extracted at the image-token
    positions.  Results are shared with :func:`compute_gmar_l2_saliency`
    via an in-process cache so both variants require only one forward pass.
    """
    return compute_gmar_variant(
        model,
        tf_inputs,
        generated_ids,
        input_len,
        image_token_positions,
        cfg,
        variant="gmar_l1",
    )
