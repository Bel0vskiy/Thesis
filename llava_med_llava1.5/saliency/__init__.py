"""Saliency-method registry.

Usage::

    from saliency import get_saliency_fn
    compute = get_saliency_fn("attention")   # or "gradcam"
    maps = compute(model, tf_inputs, generated_ids, input_len,
                   image_token_positions, cfg)
"""

from __future__ import annotations
from typing import Callable

import numpy as np


def _normalize_maps(raw_maps: "dict[int, np.ndarray]") -> "dict[int, np.ndarray]":
    """Per-token min-max normalization without cross-token coupling."""
    out: "dict[int, np.ndarray]" = {}
    for pos, raw in raw_maps.items():
        sal = raw.astype(np.float32, copy=True)
        lo = float(sal.min())
        hi = float(sal.max())
        if hi - lo > 1e-9:
            sal = (sal - lo) / (hi - lo)
        else:
            sal = np.zeros_like(sal, dtype=np.float32)
        out[pos] = sal.astype(np.float32)
    return out


def suppress_sinks(
    raw_maps: "dict[int, np.ndarray]",
    consistency_threshold: float = 0.75,
    floor_percentile: float = 10.0,
) -> "dict[int, np.ndarray]":
    """Detect and suppress attention-sink regions before per-token normalisation.

    A spatial location is a *sink* when it maintains a consistently high
    activation value across **all** generated tokens for the sample.  Such
    locations are structurally biased (e.g. the top row of CLIP patch tokens
    that corresponds to the scanner gantry edge, or residual BOS artefacts)
    rather than semantically meaningful.

    Detection
    ---------
    Pixel (i, j) is labelled a sink when its value never drops below
    ``consistency_threshold`` times its own per-pixel maximum across every
    token map.  Only pixels whose peak exceeds 10 % of the global maximum are
    considered (avoids mislabelling near-zero noise pixels).

    Suppression
    -----------
    1. Detected sink pixels are **zeroed** before any further processing.
    2. The per-pixel floor used for subtraction is the ``floor_percentile``-th
       percentile across token maps (default 10th).  This is more robust than
       the strict minimum, which can be pulled down by a single aberrant token
       and leave residual sink mass in every other map.

    Parameters
    ----------
    raw_maps : dict mapping generated-token position → raw [g, g] saliency map
    consistency_threshold : fraction in (0, 1).  Higher → stricter sink
        detection (fewer pixels suppressed).  Default 0.75 means a pixel is a
        sink if it never falls below 75 % of its own peak.
    floor_percentile : percentile in [0, 100] used as the per-pixel floor.
        Default 10 uses the 10th-percentile across token maps per pixel.

    Returns
    -------
    dict of the same structure as *raw_maps*, with sinks zeroed and each map
    min-floor-subtracted and normalised to [0, 1].
    """
    if not raw_maps:
        return {}

    all_raw = np.stack(list(raw_maps.values()), axis=0)   # [N, g, g]

    # ── sink detection ────────────────────────────────────────────────────
    pixel_min = all_raw.min(axis=0)   # [g, g] — minimum value across all tokens
    pixel_max = all_raw.max(axis=0)   # [g, g] — maximum value across all tokens

    # Only consider pixels whose peak is meaningfully above background noise
    # (avoids labelling near-zero pixels as sinks).
    global_max = float(pixel_max.max())
    nontrivial = pixel_max > 0.1 * global_max

    # A pixel is a sink if it never drops below `consistency_threshold` of
    # its own peak across all token maps.
    sink_mask = nontrivial & (pixel_min >= consistency_threshold * pixel_max)

    # ── robust floor (percentile rather than strict min) ──────────────────
    floor = np.percentile(all_raw, floor_percentile, axis=0)  # [g, g]

    out: "dict[int, np.ndarray]" = {}
    for pos, raw in raw_maps.items():
        sal = raw - floor
        sal = np.clip(sal, 0.0, None)
        sal[sink_mask] = 0.0          # zero out detected sink regions
        hi = float(sal.max())
        if hi > 1e-9:
            sal = sal / hi
        else:
            sal = np.zeros_like(raw)
        out[pos] = sal.astype(np.float32)

    return out


def postprocess_maps(
    raw_maps: "dict[int, np.ndarray]",
    cfg=None,
) -> "dict[int, np.ndarray]":
    """Postprocess saliency maps with configurable sink suppression.

    Default behavior for LLaVA-Med is method-preserving normalization only;
    sink suppression is opt-in via config.
    """
    if not raw_maps:
        return {}

    if cfg is None:
        return _normalize_maps(raw_maps)

    enable_sink = bool(getattr(cfg, "enable_sink_suppression", False))
    if not enable_sink:
        return _normalize_maps(raw_maps)

    consistency = float(getattr(cfg, "sink_consistency_threshold", 0.75))
    floor_pct = float(getattr(cfg, "sink_floor_percentile", 10.0))
    return suppress_sinks(
        raw_maps,
        consistency_threshold=consistency,
        floor_percentile=floor_pct,
    )

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator that registers a saliency function under *name*."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_saliency_fn(name: str) -> Callable:
    # Force the sub-modules to register themselves
    from . import attention as _a, gradcam as _g  # noqa: F401
    from . import gmar_l1 as _gm1, gmar_l2 as _gm2  # noqa: F401
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown saliency method '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]
