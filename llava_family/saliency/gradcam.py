"""
Grad-CAM saliency for LLaVA vision-language models.

Hooks into ``model.multi_modal_projector`` to capture the image
embeddings that are fed to the language model.  For every generated token
a backward pass computes the gradient of that token's logit w.r.t. the
projector output, and position-sensitive Grad-CAM is applied:

    L_i = ReLU(Σ_c  ∂y/∂A_{i,c} · A_{i,c})

The result is a grid × grid saliency map (one value per image token).
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


# ─────────────────────────────────────────────────────────────────────────
# Forward hook that captures – and makes differentiable – the projector
# output.
# ─────────────────────────────────────────────────────────────────────────
class _ProjectorHook:
    def __init__(self):
        self.activation: torch.Tensor | None = None
        self._handle = None

    def __call__(self, module, inp, out):
        out = out.detach().requires_grad_(True)
        out.retain_grad()
        self.activation = out
        return out

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
    """Return ``{abs_position: grid×grid numpy saliency map}``.

    One forward pass + *N* backward passes (with ``retain_graph``).
    """
    total_len = generated_ids.shape[1]
    grid = cfg.image_token_grid

    # Find the multi-modal projector (attribute path varies across
    # transformers versions and LLaVA variants).
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
        outputs = model(**_prepare_forward_kwargs(model, tf_inputs))
        logits = outputs.logits                     # [1, logit_len, V]
        activation = hook.activation                # [1, N_img, hidden]

        if activation is None:
            raise RuntimeError(
                "Grad-CAM hook captured nothing. "
                "Ensure pixel_values is present in tf_inputs."
            )

        # LLaVA-Med expands -200 → 576 patch tokens; output logits are expanded.
        logit_len = int(logits.shape[1])
        expansion_offset = logit_len - total_len   # 575 for LLaVA-Med, 0 otherwise

        gen_positions = list(range(input_len, total_len))
        saliency_maps: dict[int, np.ndarray] = {}

        for i, pos in enumerate(
            tqdm(gen_positions, desc="  gradcam saliency", leave=False)
        ):
            target_id = generated_ids[0, pos].item()
            logit_idx = pos - 1 + expansion_offset
            if logit_idx < 0 or logit_idx >= logit_len:
                continue
            score = logits[0, logit_idx, target_id]

            retain = i < len(gen_positions) - 1
            (grad,) = torch.autograd.grad(
                score,
                activation,
                retain_graph=retain,
                create_graph=False,
            )

            cam = (grad * activation.detach()).sum(dim=-1)   # [1, N_img]
            cam = torch.relu(cam)[0]                         # [N_img]

            sal = cam.cpu().float().numpy().reshape(grid, grid)
            lo = sal.min()
            hi = sal.max()
            if hi - lo > 1e-9:
                sal = (sal - lo) / (hi - lo)
            saliency_maps[pos] = sal.astype(np.float32)

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return saliency_maps
