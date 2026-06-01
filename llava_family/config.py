"""
Config class for the LLaVA-Med saliency evaluation pipeline.

Architecture notes
------------------
LLaVA 1.5 uses CLIP ViT-L/14 @ 336 px → 336/14 = 24 patches per side
→ 24 × 24 = 576 image tokens (CLS token is dropped).
An MLP projector maps these to the LM hidden dimension without spatial
pooling, so all 576 tokens enter the language model.

The original LLaVA-Med (v1.0) used 224 px → 16 × 16 = 256 tokens.
Set ``image_token_grid = 16`` if you are using the v1.0 checkpoint.

LLaMA / Mistral back-bones use standard full-sequence attention in every
layer (no global / local alternation like Gemma-3), so
``global_attn_layers`` simply lists a representative subset of layers.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────────
    model_id: str = "llava-hf/llava-1.5-7b-hf"     # replace with your LLaVA-Med checkpoint
    load_in_4bit: bool = True
    attn_implementation: str = "eager"               # required for output_attentions

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_name: str = "eltorio/ROCOv2-radiology"
    dataset_split: str = "train"
    image_column: str = "image"
    caption_column: str = "caption"
    num_samples: int = 50
    max_new_tokens: int = 100
    prompt: str = (
        "Write a single-sentence radiology caption for this medical image." 
        
    )
    #"Write a single-sentence radiology caption for this medical "
    # "image. "
    # ── Saliency methods ─────────────────────────────────────────────────
    methods: List[str] = field(
        default_factory=lambda: ["attention", "gradcam", "gmar_l1", "gmar_l2"]
    )

    # LLaVA-Med-only saliency postprocess controls.
    # Keep sink suppression OFF by default because it is method-agnostic and
    # can collapse inter-method ranking (attention/GMAR/Grad-CAM) when the
    # model already has low spatial contrast.
    enable_sink_suppression: bool = False
    sink_consistency_threshold: float = 0.75
    sink_floor_percentile: float = 10.0

    attention_layer_strategy: str = "global"
    # Literature-faithful rollout keeps generated-token inheritance enabled.
    # Set True only for diagnostic runs that suppress generated-token chains.
    rollout_block_generated_tokens: bool = False
    mask_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    )
    output_dir: str = "results"
    save_visualizations: bool = True

    # ── Architecture constants (LLaVA 1.5) ──────────────────────────────
    image_token_grid: int = 24              # 24×24 = 576 image tokens
    vision_patches_per_side: int = 24
    image_token_index: int = 32000          # <image> placeholder token id

    # Representative layers for attention / GMAR rollout.
    # LLaMA-2-7B / Mistral-7B have 32 layers, all full-attention.
    global_attn_layers: List[int] = field(
        default_factory=lambda: [3, 7, 11, 15, 19, 23, 27, 31]
    )
