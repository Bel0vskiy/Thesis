"""
Model loading, inference helpers, and shared tensor-building utilities
for MedGemma / Gemma-3 vision-language models.

Public API
----------
load_model_and_processor(cfg)
    Load a model and processor from the configured checkpoint.

prepare_inputs(processor, image, prompt_text)
    Build the raw input dict from an image and a prompt string via the
    chat template.

generate_caption(model, processor, image, cfg)
    Generate a caption and return the full token sequence, decoded text,
    prompt length, and the original inputs dict.

get_token_probabilities(model, tf_inputs, generated_ids, input_len)
    Teacher-forcing forward pass; returns per-generated-token
    probabilities.

build_tf_inputs(inputs, generated_ids, input_len, pixel_values_override)
    Construct the input dict for a teacher-forcing pass, optionally
    substituting a masked image tensor.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

from .config import Config


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────
def load_model_and_processor(cfg: Config):
    """Return ``(model, processor)`` ready for inference."""
    print(f"[model] Loading {cfg.model_id} …")

    processor = AutoProcessor.from_pretrained(cfg.model_id)

    kwargs: dict = dict(
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=cfg.attn_implementation,
    )

    if cfg.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    print(f"[model] Loaded.  device_map = {getattr(model, 'hf_device_map', 'n/a')}")
    return model, processor


# ──────────────────────────────────────────────────────────────────────────
# Tokenizer helper
# ──────────────────────────────────────────────────────────────────────────
def get_tokenizer(processor):
    tok = getattr(processor, "tokenizer", None)
    return tok if tok is not None else processor


# ──────────────────────────────────────────────────────────────────────────
# Input preparation
# ──────────────────────────────────────────────────────────────────────────
def _first_real_device(model) -> torch.device:
    """Return the device of the first non-meta parameter."""
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def prepare_inputs(processor, image: Image.Image, prompt_text: str):
    """Build the raw input dict using the chat template.

    Returns ``(inputs_dict, prompt_string)``.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt_string = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(images=image, text=prompt_string, return_tensors="pt")
    return inputs, prompt_string


def move_inputs_to_device(model, inputs: dict) -> dict:
    """Move every tensor in *inputs* to the model's main device."""
    dev = _first_real_device(model)
    return {
        k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()
    }


# ──────────────────────────────────────────────────────────────────────────
# Teacher-forcing input builder
# ──────────────────────────────────────────────────────────────────────────
def build_tf_inputs(
    inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
    pixel_values_override: torch.Tensor | None = None,
) -> dict:
    """Construct a teacher-forcing input dict.

    ``generated_ids`` is the full sequence ``[prompt … generated]``.
    ``inputs`` must contain ``pixel_values`` and optionally
    ``token_type_ids`` from the original call to :func:`prepare_inputs`.
    Pass ``pixel_values_override`` to substitute a masked image for
    perturbation-based evaluation.
    """
    total_len = generated_ids.shape[1]
    device = generated_ids.device

    pv = pixel_values_override if pixel_values_override is not None else inputs["pixel_values"]

    tf: dict = {
        "input_ids": generated_ids,
        "pixel_values": pv,
        "attention_mask": torch.ones(1, total_len, dtype=torch.long, device=device),
    }

    if "token_type_ids" in inputs:
        orig_tt = inputs["token_type_ids"]          # [1, input_len]
        extra = total_len - orig_tt.shape[1]
        if extra > 0:
            pad = torch.zeros(1, extra, dtype=orig_tt.dtype, device=device)
            tf["token_type_ids"] = torch.cat([orig_tt, pad], dim=1)
        else:
            tf["token_type_ids"] = orig_tt[:, :total_len]

    return tf


# ──────────────────────────────────────────────────────────────────────────
# Caption generation
# ──────────────────────────────────────────────────────────────────────────
def generate_caption(
    model,
    processor,
    image: Image.Image,
    cfg: Config,
):
    """Generate a caption for *image*.

    Returns a four-tuple ``(generated_ids, generated_text, input_len, inputs)``:

    * ``generated_ids``  – ``[1, total_len]`` tensor including the prompt.
    * ``generated_text`` – decoded string of the generated tokens only.
    * ``input_len``      – number of prompt tokens.
    * ``inputs``         – the original processor output, moved to device.
    """
    inputs, _ = prepare_inputs(processor, image, cfg.prompt)
    inputs = move_inputs_to_device(model, inputs)

    tok = get_tokenizer(processor)
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    if pad_id is None and eos_id is not None:
        pad_id = eos_id

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    # Truncate at the first non-content token after the prompt.
    # Stops at EOS, PAD, tokens in all_special_ids, and any token that
    # decodes to an empty string after skip_special_tokens — which catches
    # model-specific control tokens not listed in all_special_ids.
    new_tokens = gen_ids[0, input_len:]
    stop_ids = {eos_id, pad_id} | set(getattr(tok, "all_special_ids", []))
    stop_ids.discard(None)

    first_stop = None
    for idx, tid in enumerate(new_tokens.tolist()):
        if tid in stop_ids:
            first_stop = idx
            break
        # Also stop on tokens that decode to nothing (hidden special tokens)
        decoded = tok.decode([tid], skip_special_tokens=True).strip()
        if not decoded:
            first_stop = idx
            break

    if first_stop is not None and first_stop > 0:
        gen_ids = gen_ids[:, : input_len + first_stop]
        new_tokens = gen_ids[0, input_len:]

    text = tok.decode(new_tokens.tolist(), skip_special_tokens=True).strip()
    return gen_ids, text, input_len, inputs


# ──────────────────────────────────────────────────────────────────────────
# Token probability extraction (teacher-forcing)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_token_probabilities(
    model,
    tf_inputs: dict,
    generated_ids: torch.Tensor,
    input_len: int,
) -> torch.Tensor:
    """Return a 1-D float tensor of per-generated-token probabilities.

    Shape ``[num_generated_tokens]``.
    """
    outputs = model(**tf_inputs)
    logits = outputs.logits                      # [1, total_len, V]
    total_len = generated_ids.shape[1]

    probs = []
    for pos in range(input_len, total_len):
        target_id = generated_ids[0, pos].item()
        p = torch.softmax(logits[0, pos - 1].float(), dim=-1)[target_id]
        probs.append(p.item())
    return torch.tensor(probs, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# Image-token positions
# ──────────────────────────────────────────────────────────────────────────
def get_image_token_positions(inputs: dict) -> torch.Tensor:
    """Return a 1-D LongTensor of *absolute* positions that hold image
    tokens in the prompt part of ``input_ids``."""
    if "token_type_ids" in inputs:
        return (inputs["token_type_ids"][0] == 1).nonzero(as_tuple=False).flatten()
    # Fallback: MedGemma uses image_token_id = 262 144
    return (inputs["input_ids"][0] == 262144).nonzero(as_tuple=False).flatten()


# ──────────────────────────────────────────────────────────────────────────
# Content-token filter (skip stop-words / sub-words / punctuation)
# ──────────────────────────────────────────────────────────────────────────
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should can could may might must need dare "
    "i me my we us our you your he him his she her it its they them their "
    "this that these those which what who whom "
    "and or but if then else when while as so because although though "
    "in on at to for of with by from into onto upon about between "
    "not no nor "
    "very much more most also just only even still already yet "
    "there here where how why all each every both few many some any "
    "than too up out off over under again further once".split()
)


def get_content_token_mask(
    tokenizer, generated_ids: torch.Tensor, input_len: int
) -> list[bool]:
    """Return a boolean list (one per generated token) that is ``True``
    for content tokens we want to evaluate and ``False`` for the rest."""
    total_len = generated_ids.shape[1]
    mask: list[bool] = []
    for pos in range(input_len, total_len):
        tid = generated_ids[0, pos].item()
        word = tokenizer.decode([tid], skip_special_tokens=True).strip().lower()

        # Skip empty, punctuation-only, or very short sub-word fragments
        if len(word) <= 1 and not word.isalpha():
            mask.append(False)
        elif word in _STOPWORDS:
            mask.append(False)
        else:
            mask.append(True)
    return mask
