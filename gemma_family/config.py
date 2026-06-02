"""
Configuration dataclass for the Gemma-family saliency evaluation pipeline.

Covers model selection, dataset parameters, saliency method choices,
perturbation settings, and architecture constants for MedGemma and
Gemma-3 vision-language models.

Architecture notes
------------------
MedGemma uses SigLIP-400M @ 896 px → 64 patches per side, producing a
16×16 grid (256 tokens) after the pooling projector.

Gemma-3's 4B decoder has 34 transformer layers of which 5 use global
(full-sequence) self-attention: layers 5, 11, 17, 23, and 29.  The
remaining layers use local sliding-window attention.  Rollout and GMAR
operate over the global layers only.
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
    - **Saliency / evaluation** – method selection, perturbation mask
      ratios, output paths.
    - **Architecture constants** – image-token grid size and the subset of
      global decoder layers used for attention rollout and GMAR.
    """

    model_id: str = "google/medgemma-1.5-4b-it"
    load_in_4bit: bool = True
    attn_implementation: str = "eager"

    dataset_name: str = "eltorio/ROCOv2-radiology"
    dataset_config: str = ""          
    dataset_split: str = "train"
    image_column: str = "image"        
    caption_column: str = "caption"    
    num_samples: int = 50           
    max_new_tokens: int = 64
    prompt: str = "Write a single-sentence radiology caption for this medical image."
    methods: List[str] = field(
        default_factory=lambda: ["attention", "gradcam", "gmar_l1", "gmar_l2"]
    )
    attention_layer_strategy: str = "global"
    mask_ratios: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    )
    output_dir: str = "results"
    save_visualizations: bool = True

    # ── Architecture constants ────────────────────────────────────────────
    image_token_grid: int = 16           # pooled grid side; 16×16 = 256 tokens
    vision_patches_per_side: int = 64    # SigLIP raw patches per side before pooling
    # Global (full-sequence) attention layers in the Gemma-3 4B decoder.
    global_attn_layers: List[int] = field(
        default_factory=lambda: [5, 11, 17, 23, 29]
    )
