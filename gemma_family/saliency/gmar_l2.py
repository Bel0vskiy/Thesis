"""GMAR-L2 saliency wrapper."""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config
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
    """Return ``{abs_position: grid×grid saliency map}`` using GMAR-L2.

    Head importance weights are the RMS gradient of each attention head's
    matrix w.r.t. the target token's logit (L2-norm instead of L1).  The
    weighted attention rollout is extracted at the image-token positions.
    Results are shared with :func:`compute_gmar_l1_saliency` via an
    in-process cache so both variants require only one forward pass.
    """
    return compute_gmar_variant(
        model,
        tf_inputs,
        generated_ids,
        input_len,
        image_token_positions,
        cfg,
        variant="gmar_l2",
    )
