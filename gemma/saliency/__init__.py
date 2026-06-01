"""Saliency-method registry.

Usage::

    from saliency import get_saliency_fn
    compute = get_saliency_fn("attention")   # or "gradcam"
    maps = compute(model, tf_inputs, generated_ids, input_len,
                   image_token_positions, cfg)
"""

from __future__ import annotations
import importlib
from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator that registers a saliency function under *name*."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def _load_modules(force_reload: bool = False) -> None:
    """Import (or reload) saliency submodules so decorators populate _REGISTRY."""
    module_names = (
        "saliency.attention",
        "saliency.gradcam",
        "saliency.gmar_l1",
        "saliency.gmar_l2",
    )

    for module_name in module_names:
        module = importlib.import_module(module_name)
        if force_reload:
            importlib.reload(module)


def get_saliency_fn(name: str) -> Callable:
    # Ensure registrations exist in the current module instance.
    # This is notebook-safe when users reload the parent package.
    if not _REGISTRY:
        _load_modules(force_reload=True)
    else:
        _load_modules(force_reload=False)

    if name not in _REGISTRY:
        # If a method is still missing, force one clean reload attempt.
        _load_modules(force_reload=True)

    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown saliency method '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]
