"""
Grad-CAM saliency for MedGemma vision-language models.

Hooks into the multi-modal projector to capture the image embeddings fed
to the language model. For every generated token, a backward pass computes
the gradient of that token's logit w.r.t. projector outputs.

For MedGemma, the projector is position-wise (no spatial mixing), so we
use position-sensitive attribution:

    L_i = ReLU(Σ_c (∂y / ∂A_{i,c}) · A_{i,c})

This preserves per-image-token signal that can be washed out by classical
channel-pooled Grad-CAM.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..config import Config
from . import register


# ─────────────────────────────────────────────────────────────────────────
# Forward hook that captures – and makes differentiable – the projector
# output.
# ─────────────────────────────────────────────────────────────────────────
class _ProjectorHook:
    def __init__(self):
        self.activation: torch.Tensor | None = None
        self._handle = None

    # --- hook callback ---------------------------------------------------
    def __call__(self, module, inp, out):
        # out: [B, 256, text_hidden_size]
        out = out.detach().requires_grad_(True)
        # retain_grad so we can read .grad after backward
        out.retain_grad()
        self.activation = out
        return out                      # replaced in the graph

    # --- lifecycle -------------------------------------------------------
    def register(self, module):
        self._handle = module.register_forward_hook(self)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────
@register("gradcam")
def compute_gradcam_saliency(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    image_token_positions: torch.Tensor,   # unused but kept for API uniformity
    cfg: Config,
) -> dict[int, np.ndarray]:
    """Return ``{abs_position: 16×16 numpy saliency map}``.

    One forward pass + *N* backward passes (with ``retain_graph``).
    """
    total_len = generated_ids.shape[1]
    grid = cfg.image_token_grid

    # Find the multi-modal projector (attribute path can vary across
    # transformers / checkpoint revisions).
    projector = None
    for path in ("multi_modal_projector", "model.multi_modal_projector",
                 "model.mm_projector", "mm_projector"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            projector = obj
            break
        except AttributeError:
            continue
    if projector is None:
        raise AttributeError(
            "Cannot locate the multi-modal projector module on the model. "
            f"Top-level children: {[n for n, _ in model.named_children()]}"
        )

    hook = _ProjectorHook().register(projector)

    try:
        # ── single forward pass (gradients ON for the hooked tensor) ─────
        outputs = model(**tf_inputs)
        logits = outputs.logits                     # [1, total_len, V]
        activation = hook.activation                # [1, 256, hidden]

        if activation is None:
            raise RuntimeError(
                "Grad-CAM hook captured nothing. "
                "Ensure pixel_values is present in tf_inputs."
            )

        gen_positions = list(range(input_len, total_len))
        saliency_maps: dict[int, np.ndarray] = {}

        for i, pos in enumerate(
            tqdm(gen_positions, desc="  gradcam saliency", leave=False)
        ):
            target_id = generated_ids[0, pos].item()
            score = logits[0, pos - 1, target_id]   # logit before softmax

            retain = i < len(gen_positions) - 1
            (grad,) = torch.autograd.grad(
                score,
                activation,
                retain_graph=retain,
                create_graph=False,
            )
            # grad, activation: [1, 256, hidden]

            # Position-sensitive Grad-CAM (gradient x activation per image
            # token), then sum channels.
            cam = (grad * activation.detach()).sum(dim=-1)   # [1, N_img]
            cam = torch.relu(cam)[0]                         # [N_img]

            cam_np = cam.cpu().float().numpy().reshape(grid, grid)

            lo, hi = cam_np.min(), cam_np.max()
            if hi - lo > 1e-9:
                cam_np = (cam_np - lo) / (hi - lo)

            saliency_maps[pos] = cam_np.astype(np.float32)

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return saliency_maps
