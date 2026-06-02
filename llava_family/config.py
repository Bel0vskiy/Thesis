"""
Configuration dataclass for the LLaVA-family saliency evaluation pipeline.

Covers model selection, dataset parameters, saliency method choices,
perturbation settings, and architecture constants for LLaVA 1.5 and
LLaVA-Med.

Architecture notes
------------------
LLaVA 1.5 uses CLIP ViT-L/14 @ 336 px → 24 patches per side → 24×24 = 576
image tokens (CLS dropped). An MLP projector maps these to the LM hidden
dimension without spatial pooling.

LLaVA-Med v1.0 used 224 px → 16×16 = 256 tokens; set
``image_token_grid = 16`` for that checkpoint.

LLaMA-2 / Mistral backbones use standard full-sequence self-attention in
every layer, so ``global_attn_layers`` is simply a representative subset
of layer indices.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Unified configuration for model loading, dataset loading, saliency
    computation, and perturbation-based evaluation.

    Fields are grouped into four logical sections:

    - **Model** – checkpoint path, quantisation, attention implementation.
    - **Dataset** – source, split, column names, sample count, generation
      parameters.
    - **Saliency / evaluation** – method selection, sink-suppression options,
      perturbation mask ratios, output paths.
    - **Architecture constants** – image-token grid size and the subset of
      decoder layers used for attention rollout and GMAR.
    """

    # ── Model ────────────────────────────────────────────────────────────
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    load_in_4bit: bool = True
    attn_implementation: str = "eager"  # must be "eager" to obtain output_attentions

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_name: str = "eltorio/ROCOv2-radiology"
    dataset_split: str = "train"
    image_column: str = "image"
    caption_column: str = "caption"
    num_samples: int = 50
    max_new_tokens: int = 100
    prompt: str = "Write a single-sentence radiology caption for this medical image."

    # ── Saliency / evaluation ─────────────────────────────────────────────
    methods: List[str] = field(
        default_factory=lambda: ["attention", "gradcam", "gmar_l1", "gmar_l2"]
    )

    # Attention-sink suppression for saliency post-processing.
    enable_sink_suppression: bool = False
    sink_consistency_threshold: float = 0.75
    sink_floor_percentile: float = 10.0

    attention_layer_strategy: str = "global"
    rollout_block_generated_tokens: bool = False
    mask_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    )
    output_dir: str = "results"
    save_visualizations: bool = True

    # ── Architecture constants ────────────────────────────────────────────
    image_token_grid: int = 24          # patches per side; 24×24 = 576 tokens (LLaVA 1.5)
    vision_patches_per_side: int = 24
    image_token_index: int = 32000      # token ID of the <image> placeholder

    # Subset of decoder layers used for attention rollout and GMAR.
    # LLaMA-2-7B / Mistral-7B have 32 full-attention layers.
    global_attn_layers: List[int] = field(
        default_factory=lambda: [3, 7, 11, 15, 19, 23, 27, 31]
    )
