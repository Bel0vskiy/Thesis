"""
Model loading, inference helpers, and shared tensor-building utilities
for LLaVA / LLaVA-Med vision-language models.

Public API
----------
load_model_and_processor(cfg)
    Load a model and processor/tokenizer from the configured checkpoint.
    For LLaVA-Med checkpoints, uses the official Microsoft ``llava`` package
    loader when available; falls back to the Hugging Face
    ``LlavaForConditionalGeneration`` path otherwise.

prepare_inputs(processor, image, prompt_text, model)
    Build the raw input dict from an image and a prompt string.

generate_caption(model, processor, image, cfg)
    Generate a caption and return the full token sequence, decoded text,
    prompt length, and the original inputs dict.

get_token_probabilities(model, tf_inputs, generated_ids, input_len)
    Teacher-forcing forward pass; returns per-generated-token probabilities.

build_tf_inputs(inputs, generated_ids, input_len, pixel_values_override)
    Construct the input dict for a teacher-forcing pass, optionally
    substituting a masked image tensor.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from config import Config


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────
def load_model_and_processor(cfg: Config):
    """Return ``(model, processor)`` ready for inference."""
    print(f"[model] Loading {cfg.model_id} …")

    # True LLaVA-Med path via the official llava package.
    if "llava-med" in cfg.model_id.lower():
        try:
            from llava.constants import IMAGE_TOKEN_INDEX
            from llava.mm_utils import get_model_name_from_path
            from llava.model.builder import load_pretrained_model

            model_name = get_model_name_from_path(cfg.model_id)
            try:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    cfg.model_id,
                    None,
                    model_name,
                    load_8bit=False,
                    load_4bit=bool(cfg.load_in_4bit),
                    device_map="auto",
                )
            except TypeError:
                tokenizer, model, image_processor, _ = load_pretrained_model(
                    cfg.model_id,
                    None,
                    model_name,
                )

            model.eval()
            cfg.image_token_index = int(IMAGE_TOKEN_INDEX)

            class LlavaMedProcessorAdapter:
                def __init__(self, tokenizer, image_processor, name_or_path: str):
                    self.tokenizer = tokenizer
                    self.image_processor = image_processor
                    self.name_or_path = name_or_path
                    self._is_llava_med_adapter = True

            processor = LlavaMedProcessorAdapter(tokenizer, image_processor, cfg.model_id)
            print(
                f"[model] Loaded OFFICIAL LLaVA-Med class={model.__class__.__name__} "
                f"(image_token_index={cfg.image_token_index})"
            )
            return model, processor
        except Exception as e:
            msg = str(e)
            # After a CUDA device-side assert, the runtime CUDA context is
            # poisoned. Falling back to a different loader in the same kernel
            # usually fails with confusing secondary errors.
            if "device-side assert" in msg.lower() or "cuda error" in msg.lower():
                raise RuntimeError(
                    "Official LLaVA-Med loader hit a CUDA device-side assert. "
                    "Restart the runtime and reload the model from scratch. "
                    f"Original error: {e}"
                )

            print(f"[model] Official LLaVA-Med loader unavailable, falling back to HF path: {e}")

    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)

    kwargs: dict = dict(
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation=cfg.attn_implementation,
        trust_remote_code=True,
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

        # In some environments, bitsandbytes may be discoverable as a module
        # but missing package metadata, which causes HF internals to raise
        # PackageNotFoundError even for non-quantized loading.  Patch the
        # availability flags to avoid a spurious failure.
        try:
            import importlib.metadata as _ilm
            import transformers.modeling_utils as _mu
            import transformers.utils.import_utils as _iu

            try:
                _ilm.version("bitsandbytes")
            except _ilm.PackageNotFoundError:
                _mu.is_bitsandbytes_available = lambda: False
                _iu.is_bitsandbytes_available = lambda: False
        except Exception:
                # Best-effort patch; if it fails, the main load path will
                # surface a clear traceback.
                pass

    # Compatibility path: this works across more runtime combinations
    # (including environments where llava_mistral is not auto-registered).
    model = LlavaForConditionalGeneration.from_pretrained(cfg.model_id, **kwargs)
    model.eval()

    # Auto-detect image_token_index from model config when available
    if hasattr(model.config, "image_token_index"):
        cfg.image_token_index = model.config.image_token_index

    print(
        f"[model] Loaded class={model.__class__.__name__} "
        f"(config.model_type={getattr(model.config, 'model_type', 'n/a')}). "
        f"device_map={getattr(model, 'hf_device_map', 'n/a')}"
    )
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


def _candidate_vision_towers(model) -> list[str]:
    """Return possible vision tower identifiers from model config."""
    if model is None:
        return []

    cfg = getattr(model, "config", None)
    if cfg is None:
        return []

    candidates = []
    for attr in ("vision_tower", "mm_vision_tower", "vision_model_name_or_path"):
        v = getattr(cfg, attr, None)
        if isinstance(v, str) and v:
            candidates.append(v)
        elif isinstance(v, (list, tuple)) and v:
            first = v[0]
            if isinstance(first, str) and first:
                candidates.append(first)

    # Preserve order while deduplicating.
    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def prepare_inputs(processor, image: Image.Image, prompt_text: str, model=None):
    """Build the raw input dict for a model forward pass.

    Tries the HF chat-template API first, then falls back to the manual
    Vicuna-style prompt format used by many LLaVA checkpoints.  Also handles
    processors that return text-only outputs by fetching image tensors from
    the image processor directly.

    Returns ``(inputs_dict, prompt_string)``.
    """
    if getattr(processor, "_is_llava_med_adapter", False):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        prompt_string = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: {DEFAULT_IMAGE_TOKEN}\n{prompt_text} ASSISTANT:"
        )
        input_ids = tokenizer_image_token(
            prompt_string,
            processor.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        image_tensor = process_images([image], processor.image_processor, model.config if model is not None else None)
        if isinstance(image_tensor, (list, tuple)):
            image_tensor = image_tensor[0]
        if torch.is_tensor(image_tensor) and image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "images": image_tensor,
        }
        return inputs, prompt_string

    prompt_string = None

    # Attempt 1: chat-template (works with llava-hf/* checkpoints)
    try:
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
    except Exception:
        pass

    # Attempt 2: manual Vicuna-style prompt
    if prompt_string is None:
        prompt_string = f"USER: <image>\n{prompt_text}\nASSISTANT:"

    # Try multiple processor signatures across transformers/model variants.
    attempts = [
        lambda: processor(images=image, text=prompt_string, return_tensors="pt"),
        lambda: processor(images=[image], text=prompt_string, return_tensors="pt"),
        lambda: processor(text=prompt_string, images=image, return_tensors="pt"),
        lambda: processor(text=prompt_string, images=[image], return_tensors="pt"),
    ]

    inputs = None
    for fn in attempts:
        try:
            cand = fn()
            if "input_ids" in cand:
                inputs = cand
                try:
                    _ = get_vision_input_key(cand)
                    break
                except Exception:
                    # Keep trying; a later variant may include the vision tensor.
                    pass
        except Exception:
            pass

    if inputs is None:
        # Last-resort text inputs.
        inputs = processor(text=prompt_string, return_tensors="pt")

    # Some checkpoints/processors produce text tensors only. In that case,
    # build vision tensors directly and merge.
    try:
        _ = get_vision_input_key(inputs)
    except Exception:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            # Fallback for checkpoints where AutoProcessor resolves to tokenizer-only.
            candidate_ids = [getattr(processor, "name_or_path", None)] + _candidate_vision_towers(model)
            for repo_id in candidate_ids:
                if not repo_id:
                    continue
                try:
                    image_processor = AutoImageProcessor.from_pretrained(
                        repo_id,
                        trust_remote_code=True,
                    )
                    break
                except Exception:
                    image_processor = None

        if image_processor is not None:
            try:
                img_inputs = image_processor(images=image, return_tensors="pt")
            except Exception:
                img_inputs = image_processor(images=[image], return_tensors="pt")

            for k, v in img_inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v

            # Normalize common aliases so downstream code has a stable key.
            if "pixel_values" not in inputs:
                for alias in ("pixel_values_images", "image_pixels", "images"):
                    if alias in inputs and torch.is_tensor(inputs[alias]):
                        inputs["pixel_values"] = inputs[alias]
                        break

    return inputs, prompt_string


def move_inputs_to_device(model, inputs: dict) -> dict:
    """Move every tensor in *inputs* to the model's main device."""
    dev = _first_real_device(model)
    try:
        return {
            k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()
        }
    except Exception as e:
        msg = str(e).lower()
        if "device-side assert" in msg or "cuda error" in msg:
            raise RuntimeError(
                "CUDA context is in an error state (device-side assert). "
                "Restart the runtime and reload the model from scratch. "
                f"Original error: {e}"
            )
        raise


def get_vision_input_key(inputs: dict) -> str:
    """Return the key used for image tensors in *inputs*.

    Different LLaVA-family checkpoints expose image tensors under different
    keys. Raises ``KeyError`` with a descriptive message if no recognised
    vision key is present.
    """
    for key in (
        "pixel_values",
        "images",
        "image_patches",
        "pixel_values_images",
        "image_pixels",
    ):
        if key in inputs and torch.is_tensor(inputs[key]):
            return key
    raise KeyError(
        "No vision tensor found in inputs. Expected one of: "
        "'pixel_values', 'images', 'image_patches', 'pixel_values_images', 'image_pixels'. "
        f"Available keys: {sorted(inputs.keys())}"
    )


def get_vision_tensor(inputs: dict) -> torch.Tensor:
    """Return the vision tensor from an inputs dict."""
    return inputs[get_vision_input_key(inputs)]


def _prepare_generate_inputs(model, inputs: dict) -> dict:
    """Normalise multimodal inputs before generation.

    Ensures the vision tensor is keyed as ``pixel_values``, removes
    conflicting aliases, adds a batch dimension if missing, and injects
    an image-token placeholder into ``input_ids`` when none is present.
    """
    out = dict(inputs)

    vision_key = get_vision_input_key(out)
    pv = out[vision_key]
    if torch.is_tensor(pv) and pv.dim() == 3:
        pv = pv.unsqueeze(0)
    if torch.is_tensor(pv) and pv.dim() == 4 and pv.shape[1] != 3 and pv.shape[-1] == 3:
        # Some processors can return NHWC; model expects NCHW.
        pv = pv.permute(0, 3, 1, 2).contiguous()

    out["pixel_values"] = pv
    for alias in ("images", "image_patches", "pixel_values_images", "image_pixels"):
        if alias != "pixel_values" and alias in out:
            del out[alias]

    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)

    # If input_ids already contains a negative placeholder (e.g., -200 for official
    # LLaVA-Med), the image marker is already present — skip injection entirely.
    # Injecting an extra token here corrupts the prompt with a spurious <unk> (ID=0).
    if "input_ids" in out and torch.is_tensor(out["input_ids"]):
        if bool((out["input_ids"] < 0).any().item()):
            return out

    vocab_size = None
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            vocab_size = int(emb.weight.shape[0])
    except Exception:
        vocab_size = None

    # Build candidate image token IDs from config and tokenizer, then ensure
    # at least one is present in input_ids.
    candidate_ids = []
    if image_token_index is not None:
        candidate_ids.append(int(image_token_index))

    tok = getattr(model, "_saliency_tokenizer", None)
    if tok is not None:
        for marker in ("<image>", "<image_token>", "<img>"):
            try:
                tid = int(tok.convert_tokens_to_ids(marker))
                if tid >= 0:
                    candidate_ids.append(tid)
            except Exception:
                pass

        try:
            add_tokens = list(getattr(tok, "additional_special_tokens", []) or [])
            add_ids = list(getattr(tok, "additional_special_tokens_ids", []) or [])
            for t, tid in zip(add_tokens, add_ids):
                if isinstance(t, str) and "image" in t.lower():
                    candidate_ids.append(int(tid))
        except Exception:
            pass

    # Deduplicate and keep in-vocab candidates.
    dedup = []
    seen = set()
    for tid in candidate_ids:
        if tid in seen:
            continue
        if tid < 0:
            continue
        if vocab_size is not None and tid >= vocab_size:
            continue
        dedup.append(tid)
        seen.add(tid)

    if dedup and "input_ids" in out:
        input_ids = out["input_ids"]
        if torch.is_tensor(input_ids):
            num_img_toks = 0
            for tid in dedup:
                num_img_toks += int((input_ids == int(tid)).sum().item())
            if num_img_toks == 0:
                chosen_tid = int(dedup[0])
                bsz = input_ids.shape[0]
                img_tok = torch.full(
                    (bsz, 1), chosen_tid, dtype=input_ids.dtype, device=input_ids.device
                )
                if input_ids.shape[1] >= 1:
                    input_ids = torch.cat([input_ids[:, :1], img_tok, input_ids[:, 1:]], dim=1)
                else:
                    input_ids = torch.cat([img_tok, input_ids], dim=1)
                out["input_ids"] = input_ids

                if "attention_mask" in out and torch.is_tensor(out["attention_mask"]):
                    am = out["attention_mask"]
                    ones = torch.ones((bsz, 1), dtype=am.dtype, device=am.device)
                    if am.shape[1] >= 1:
                        am = torch.cat([am[:, :1], ones, am[:, 1:]], dim=1)
                    else:
                        am = torch.cat([ones, am], dim=1)
                    out["attention_mask"] = am

    return out


def _validate_generate_inputs(model, inputs: dict) -> None:
    """Raise a descriptive error if multimodal inputs are invalid.

    Checks that ``input_ids`` and ``pixel_values`` are present and that all
    token IDs are within the model vocabulary.  The LLaVA-family negative
    placeholder (commonly -200) is explicitly allowed.
    """
    if "input_ids" not in inputs or not torch.is_tensor(inputs["input_ids"]):
        raise RuntimeError("Missing tensor input_ids before generation.")

    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, "weight"):
        raise RuntimeError("Model input embeddings are not available.")

    vocab_size = int(emb.weight.shape[0])
    input_ids = inputs["input_ids"]

    # LLaVA-family models can use a negative placeholder token (commonly -200)
    # for image slots before multimodal fusion. Treat these IDs as valid.
    allowed_negative_ids = {-200}
    image_token_index = getattr(getattr(model, "config", None), "image_token_index", None)
    if isinstance(image_token_index, int) and image_token_index < 0:
        allowed_negative_ids.add(int(image_token_index))

    bad_negative = (input_ids < 0)
    for neg_id in allowed_negative_ids:
        bad_negative = bad_negative & (input_ids != neg_id)

    has_bad_negative = bool(bad_negative.any().item())
    has_bad_positive = bool((input_ids >= vocab_size).any().item())
    if has_bad_negative or has_bad_positive:
        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        raise RuntimeError(
            "input_ids contain unsupported token IDs: "
            f"min={min_id}, max={max_id}, vocab_size={vocab_size}, "
            f"allowed_negative_ids={sorted(allowed_negative_ids)}."
        )

    pv = inputs.get("pixel_values", None)
    if pv is None or not torch.is_tensor(pv):
        raise RuntimeError("Missing tensor pixel_values before generation.")
    if pv.dim() != 4:
        raise RuntimeError(f"pixel_values must be 4D [B,C,H,W], got shape={tuple(pv.shape)}")
    if not torch.is_floating_point(pv):
        inputs["pixel_values"] = pv.float()


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
    ``inputs`` must contain the vision tensor from the original
    :func:`prepare_inputs` call.  Pass ``pixel_values_override`` to
    substitute a masked image for perturbation-based evaluation.
    """
    total_len = generated_ids.shape[1]
    device = generated_ids.device

    # For official LLaVA-Med: prefer the 'images' key over 'pixel_values' so the
    # model's forward() triggers multimodal fusion (expanding -200 → 576 patches).
    # generate_caption restores 'images' in inputs for this purpose.
    if "images" in inputs and torch.is_tensor(inputs["images"]):
        vision_key = "images"
    else:
        vision_key = get_vision_input_key(inputs)

    pv = pixel_values_override if pixel_values_override is not None else inputs[vision_key]

    tf: dict = {
        "input_ids": generated_ids,
        "attention_mask": torch.ones(1, total_len, dtype=torch.long, device=device),
    }
    tf[vision_key] = pv

    # Include image_sizes if present (needed for some LLaVA variants)
    if "image_sizes" in inputs:
        tf["image_sizes"] = inputs["image_sizes"]

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
    inputs, _ = prepare_inputs(processor, image, cfg.prompt, model=model)
    inputs = move_inputs_to_device(model, inputs)

    # Fail early with a clear message if no image tensor is present.
    try:
        _ = get_vision_input_key(inputs)
    except Exception as e:
        raise RuntimeError(
            "Processor returned text-only inputs (no image tensor). "
            "This model/processor combination cannot run vision saliency as configured. "
            f"Input keys: {sorted(inputs.keys())}. Details: {e}"
        )

    tok = get_tokenizer(processor)
    # Stash tokenizer on model for robust image-token detection.
    try:
        setattr(model, "_saliency_tokenizer", tok)
    except Exception:
        pass

    # Keep the adapter image tensor before normalization because
    # _prepare_generate_inputs may rewrite/remove alias keys.
    adapter_images = inputs.get("images", None)

    inputs = _prepare_generate_inputs(model, inputs)
    _validate_generate_inputs(model, inputs)

    # For official LLaVA-Med: restore 'images' key alongside 'pixel_values' so
    # build_tf_inputs / saliency forward passes can use the correct key.
    if getattr(processor, "_is_llava_med_adapter", False) and adapter_images is not None:
        inputs["images"] = adapter_images

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    if pad_id is None and eos_id is not None:
        pad_id = eos_id

    input_len = inputs["input_ids"].shape[1]

    image_for_generate = adapter_images if adapter_images is not None else inputs.get("pixel_values", None)
    if torch.is_tensor(image_for_generate):
        model_dev = _first_real_device(model)
        model_dtype = None
        for p in model.parameters():
            if p.device.type != "meta":
                model_dtype = p.dtype
                break
        if model_dtype is None:
            model_dtype = torch.float16

        image_for_generate = image_for_generate.to(device=model_dev)
        if torch.is_floating_point(image_for_generate):
            image_for_generate = image_for_generate.to(dtype=model_dtype)

    with torch.no_grad():
        if getattr(processor, "_is_llava_med_adapter", False):
            gen_ids = model.generate(
                inputs=inputs["input_ids"],
                images=image_for_generate,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        else:
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

    # Some official LLaVA-Med generate paths can return continuation-only
    # tokens or a sequence that does not preserve the exact prompt prefix.
    # Normalize to the expected full sequence format [prompt ... gen] so
    # teacher-forcing keeps image-conditioning tokens in place.
    prompt_ids = inputs["input_ids"]
    has_full_prefix = (
        gen_ids.shape[1] >= input_len
        and torch.equal(gen_ids[:, :input_len], prompt_ids)
    )
    if not has_full_prefix:
        gen_ids = torch.cat([prompt_ids, gen_ids], dim=1)

    # Truncate at the first non-content token after the prompt.
    new_tokens = gen_ids[0, input_len:]
    stop_ids = {eos_id, pad_id} | set(getattr(tok, "all_special_ids", []))
    stop_ids.discard(None)

    first_stop = None
    for idx, tid in enumerate(new_tokens.tolist()):
        if tid in stop_ids:
            first_stop = idx
            break

    if first_stop is not None and first_stop > 0:
        gen_ids = gen_ids[:, : input_len + first_stop]
        new_tokens = gen_ids[0, input_len:]

    text = tok.decode(new_tokens.tolist(), skip_special_tokens=True).strip()

    # Fallback: if generation is empty or extremely short, try a slightly
    # more exploratory decode to avoid degenerate outputs that make AOPC
    # uninformative.
    if len(new_tokens) <= 2 or not text:
        with torch.no_grad():
            if getattr(processor, "_is_llava_med_adapter", False):
                retry_ids = model.generate(
                    inputs=inputs["input_ids"],
                    images=image_for_generate,
                    max_new_tokens=max(int(cfg.max_new_tokens), 96),
                    min_new_tokens=12,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
            else:
                retry_ids = model.generate(
                    **inputs,
                    max_new_tokens=max(int(cfg.max_new_tokens), 96),
                    min_new_tokens=12,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

        if retry_ids.shape[1] < input_len:
            retry_ids = torch.cat([inputs["input_ids"], retry_ids], dim=1)

        retry_new = retry_ids[0, input_len:]
        retry_text = tok.decode(retry_new.tolist(), skip_special_tokens=True).strip()
        if len(retry_new) > len(new_tokens) and retry_text:
            gen_ids = retry_ids
            new_tokens = retry_new
            text = retry_text

    # For official LLaVA-Med: model.generate() may return the full sequence but
    # with -200 consumed/stripped internally.  Teacher-forcing MUST have -200 so
    # the model's multimodal fusion can expand it to 576 patch embeddings.
    # Reconstruct gen_ids = prompt_ids (has -200) + generated tokens.
    if getattr(processor, "_is_llava_med_adapter", False):
        prompt_ids_with_img = inputs["input_ids"]  # always has -200 from prepare_inputs
        gen_has_img = bool((gen_ids[0] == -200).any().item())
        prompt_has_img = bool((prompt_ids_with_img[0] == -200).any().item())
        if prompt_has_img and not gen_has_img:
            gen_toks = gen_ids[0, input_len:].unsqueeze(0)  # [1, num_generated]
            gen_ids = torch.cat([prompt_ids_with_img, gen_toks], dim=1)

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
    """Return per-generated-token probabilities via a teacher-forcing pass.

    Shape of the returned tensor: ``[num_generated_tokens]``.
    """
    call_kwargs = dict(tf_inputs)

    # Official LLaVA-Med forward expects `images` instead of `pixel_values`.
    is_official_llava = model.__class__.__module__.startswith("llava.")
    if is_official_llava and "pixel_values" in call_kwargs and "images" not in call_kwargs:
        call_kwargs["images"] = call_kwargs.pop("pixel_values")

    # Keep image tensor aligned with model compute dtype/device.
    if "images" in call_kwargs and torch.is_tensor(call_kwargs["images"]):
        img = call_kwargs["images"]
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
        call_kwargs["images"] = img

    outputs = model(**call_kwargs)
    logits = outputs.logits                      # [1, logit_len, V]
    total_len = generated_ids.shape[1]
    logit_len = int(logits.shape[1])
    vocab_size = int(logits.shape[-1])

    # LLaVA-Med expands the single -200 image placeholder to 576 patch tokens
    # during the forward pass. The output logits therefore have length
    #   logit_len = total_len + (n_patch_tokens - 1)   (e.g. +575 for 24×24)
    # whereas `pos` is an index into the *unexpanded* generated_ids sequence.
    # To read the logit that predicts the token at unexpanded position `pos`
    # we must shift by the expansion offset so that we land in the right place
    # in the expanded logit sequence.
    expansion_offset = logit_len - total_len   # 0 for text-only models, 575 for LLaVA-Med

    probs = []
    valid_positions = 0
    for pos in range(input_len, total_len):
        target_id = generated_ids[0, pos].item()
        logit_idx = pos - 1 + expansion_offset
        if logit_idx < 0 or logit_idx >= logit_len:
            probs.append(0.0)
            continue
        if not (0 <= int(target_id) < vocab_size):
            probs.append(0.0)
            continue
        valid_positions += 1
        p = torch.softmax(logits[0, logit_idx].float(), dim=-1)[target_id]
        probs.append(float(p.item()))

    if valid_positions == 0 and total_len > input_len:
        print(
            "[warn] get_token_probabilities found zero valid generated-token targets; "
            "probabilities are all zeros. This usually indicates mismatched generated_ids "
            "or missing/invalid prompt-prefix alignment."
        )
    return torch.tensor(probs, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# Image-token positions
# ──────────────────────────────────────────────────────────────────────────
def get_image_token_positions(inputs: dict, image_token_index: int = 32000) -> torch.Tensor:
    """Return a 1-D LongTensor of *absolute* positions that hold image
    tokens in the prompt part of ``input_ids``.

    LLaVA represents the image as a contiguous block of
    ``image_token_index`` values in ``input_ids``.
    """
    iids = inputs["input_ids"][0]
    positions = (iids == image_token_index).nonzero(as_tuple=False).flatten()

    # Fallback: if nothing found with the given index, try the official LLaVA-Med
    # placeholder (-200).  This handles the common case where cfg.image_token_index
    # was not updated after model load (still 32000 from the config cell).
    if len(positions) == 0 and image_token_index != -200:
        fallback_pos = (iids == -200).nonzero(as_tuple=False).flatten()
        if len(fallback_pos) > 0:
            positions = fallback_pos
            image_token_index = -200

    # Official LLaVA-Med often uses a single placeholder image token in
    # input_ids, while attention/logit paths operate over expanded patch tokens.
    # Return a slightly over-complete candidate block (n_img + 1) so downstream
    # code can drop a leading CLS-like token when present.
    if len(positions) == 1:
        try:
            pv = get_vision_tensor(inputs)
            if torch.is_tensor(pv) and pv.dim() == 4:
                h = int(pv.shape[-2])
                w = int(pv.shape[-1])
                if h % 14 == 0 and w % 14 == 0:
                    grid_h = h // 14
                    grid_w = w // 14
                    n_img = int(grid_h * grid_w)
                    if n_img > 1:
                        start = int(positions[0].item())
                        device = positions.device
                        expanded = torch.arange(start, start + n_img, device=device)
                        return expanded
        except Exception:
            pass

    return positions


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

        if len(word) <= 1 and not word.isalpha():
            mask.append(False)
        elif word in _STOPWORDS:
            mask.append(False)
        else:
            mask.append(True)
    return mask


# SentencePiece (▁, U+2581) and GPT-2 BPE (Ġ, U+0120) word-boundary prefixes.
_WORD_BOUNDARY_PREFIXES = ('\u2581', '\u0120', ' ')


def get_word_boundary_mask(
    tokenizer, generated_ids: torch.Tensor, input_len: int
) -> list[bool]:
    """Return True for the first sub-token of each surface word, False for continuations.

    Example: "ultrasound" tokenised as [▁ul, tra, sound] → [True, False, False].

    Only the first sub-token carries a meaningful saliency map for AOPC:
    subsequent sub-tokens are near-deterministic given the prefix under
    teacher-forcing, so their probability drops are negligible and dilute
    the faithfulness signal.

    Works with SentencePiece (Mistral/LLaMA — ▁ prefix) and GPT-2 BPE
    (Ġ prefix).  Falls back to decoded text when convert_ids_to_tokens is
    unavailable.
    """
    total_len = generated_ids.shape[1]
    mask: list[bool] = []
    for pos in range(input_len, total_len):
        tid = int(generated_ids[0, pos].item())

        # Prefer raw token string which retains boundary markers.
        raw = ''
        if hasattr(tokenizer, 'convert_ids_to_tokens'):
            toks = tokenizer.convert_ids_to_tokens([tid])
            if toks:
                raw = toks[0] or ''
        if not raw:
            raw = tokenizer.decode([tid], skip_special_tokens=False)

        if not mask:
            # First generated token is always word-initial.
            mask.append(True)
        elif any(raw.startswith(pfx) for pfx in _WORD_BOUNDARY_PREFIXES):
            mask.append(True)
        else:
            mask.append(False)
    return mask
