![header](https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,100:2d4a7a&height=200&text=Evaluating%20The%20Faithfulness%20of%20Medical%20VLMs&fontSize=25&fontColor=e0e8f0&animation=fadeIn)

## About

This code was produced as part of my bachelor thesis at Maastricht University in the first half of 2026. The codebase adapts ViT-based visual-saliency algorithms to medically fine-tuned VLMs and runs a per-token deletion faithfulness experiment across two model families and two datasets.

The paper produced from this code is the `thesis.pdf` file at the root of this repository — it contains all the academic details about the algorithms, adaptations, & experiments.

## Requirements

You need an NVIDIA GPU with at least 48 GB of VRAM, a good amount of free disk space for the model weights, and a solid internet connection to pull them from the Hugging Face hub. For the best experience I recommend running on an H100 or RTX 6000. You will also need a Hugging Face account with an API token.

## Navigating The Codebase

The two directories at the root of the repository contain the adaptations and experiment notebooks for two separate model families: **Gemma** (`google/medgemma-1.5-4b-it`, `google/gemma-3-4b-it`) and **LLaVA** (`llava-hf/llava-1.5-7b-hf`, `microsoft/llava-med-v1.5-mistral-7b`). Both families are evaluated on ROCOv2-radiology (medical domain) and COCO-Caption (general domain).

Below is a breakdown of the lovely files you will find in each directory.

### Notebooks

In each directory you will find one or more experiment notebooks. Gemma-3 and MedGemma share a single notebook (`deletion_medgemma.ipynb`) because both are loaded identically through the standard `transformers` `AutoModelForCausalLM` path, so only the `model_id` differs. LLaVA 1.5 and LLaVA-Med, by contrast, require separate notebooks. LLaVA-Med can only be loaded through the official Microsoft LLaVA-Med repository (`llava.model.builder.load_pretrained_model`), which ships its own model builder, tokeniser utilities, and image-token constants — hence `deletion_llava.ipynb` and `deletion_llava_med.ipynb` are kept separate.

Both `deletion_llava_med.ipynb` and `deletion_medgemma.ipynb` are designed to run on Google Colab. Each notebook contains a few optional Colab-only cells (clearly marked) that handle known runtime environment issues such as bitsandbytes CUDA mismatches and numpy ABI conflicts — leave them commented out unless you hit those specific errors.

### Saliency Algorithms

The saliency methods live in the `saliency/` subdirectory of each model family and are initialised as a Python module. All four methods receive the same inputs — the model, the token-forced inputs, the generated token ids, the input length, the image token positions, and the config — and return a per-token saliency heatmap of shape `(grid_size, grid_size)`.

## Datasets

Two datasets are used in this experiment, toggled via the `USE_COCO` flag in the config cell of the notebooks.

- **ROCOv2-radiology** (`eltorio/ROCOv2-radiology`, `train` split) — the primary dataset. Contains radiology images paired with expert-written captions sourced from biomedical literature. This is the medically relevant evaluation: the model is prompted to produce a radiology caption and the faithfulness of its saliency maps is measured against that output.

- **COCO-Caption** (`lmms-lab/COCO-Caption`, `val` split) — a general-domain reference dataset of natural images with crowd-sourced captions. Included to allow comparison of faithfulness scores between medical and non-medical visual content.

### Other Files

Each directory contains the same supporting modules:

- **`config.py`** — Dataclass holding all hyperparameters: model id, dataset, number of samples, saliency methods, mask ratios, output directory, and architecture constants (image token grid size, global attention layers, etc.). The notebook config cells override the defaults directly on a `Config` instance.
- **`dataset.py`** — Loads and shuffles samples from the Hugging Face hub using the dataset settings in the config. Returns a list of dicts with `image`, `caption`, and `id` keys.
- **`model_utils.py`** — Handles model & processor loading (including the official LLaVA-Med loader path), caption generation, teacher-forcing input construction, token probability extraction, image token position lookup, and content token masking.
- **`evaluation.py`** — Implements the perturbation loop. The main function is `evaluate_faithfulness_per_token`, which runs one forward pass per generated token per mask ratio, zeroing out the top-k image patches according to the saliency map and measuring the drop in that token's probability. The aggregate metric is AOPC (Area Over Perturbation Curve) — the mean probability drop across mask ratios. A random baseline (`evaluate_faithfulness_random`) masks patches in random order and serves as the lower-bound reference.
- **`visualization.py`** — Saves saliency grid figures, per-method comparison figures, and perturbation curves for each sample when `cfg.save_visualizations = True`.
- **`run.py`** — CLI entry point wrapping the same experiment loop as the notebooks. Useful for running experiments on a remote machine without Jupyter.