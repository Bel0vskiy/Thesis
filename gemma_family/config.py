"""
config class for setting hyperparameters and options for the saliency evaluation pipeline
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    model_id: str = "google/medgemma-1.5-4b-it"
    load_in_4bit: bool = True
    attn_implementation: str = "eager"

    dataset_name: str = "eltorio/ROCOv2-radiology"
    dataset_config: str = ""           # HuggingFace subset/config name (e.g. '2017_captions' for COCO)
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

    #Architecture constants
    image_token_grid: int = 16          # 16×16 tokens after projector pooling
    vision_patches_per_side: int = 64   # 64×64 patches before pooling
    global_attn_layers: List[int] = field(
        default_factory=lambda: [5, 11, 17, 23, 29]
    )
