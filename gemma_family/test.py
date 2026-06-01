import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-1.5-4b-it"
IMAGE_PATH = "lungcancer2.png"


def move_inputs_to_model_device(model, inputs):
    # device_map="auto" -> pick a real device
    dev = None
    for p in model.parameters():
        if p.device.type != "meta":
            dev = p.device
            break
    if dev is None:
        dev = torch.device("cpu")
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in inputs.items()}, dev


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tok = getattr(processor, "tokenizer", None)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")

    question = "Describe this image and any notable findings."

    # ---- IMPORTANT: prefer proper multimodal chat content structure ----
    prompt = None
    inputs = None

    if hasattr(processor, "apply_chat_template"):
        # Many Gemma3-derived multimodal models expect content list with image + text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    else:
        # Fallback if no chat template
        prompt = f"<start_of_image> {question}"
        inputs = processor(images=image, text=prompt, return_tensors="pt")

    inputs, dev = move_inputs_to_model_device(model, inputs)

    pad_id = tok.pad_token_id if tok and tok.pad_token_id is not None else None
    eos_id = tok.eos_token_id if tok and tok.eos_token_id is not None else None

    # If pad is missing, set it to eos (helps some setups)
    if tok is not None and tok.pad_token_id is None and tok.eos_token_id is not None:
        pad_id = tok.eos_token_id

    # ---- generate ----
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=None,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
        else:
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

    # ---- DEBUG PRINTS ----
    input_len = inputs["input_ids"].shape[1]
    tail = gen_ids[0, input_len:]

    print("\n=== PROMPT (string passed to processor) ===\n")
    print(prompt)

    print("\n=== TAIL TOKEN IDS ===\n")
    print(tail.tolist())

    print("\n=== FULL DECODE (keep specials) ===\n")
    if tok is not None:
        print(tok.decode(gen_ids[0].tolist(), skip_special_tokens=False))
    else:
        print(processor.decode(gen_ids[0].tolist(), skip_special_tokens=False))

    print("\n=== ANSWER ONLY (tail, keep specials) ===\n")
    if tok is not None:
        print(tok.decode(tail.tolist(), skip_special_tokens=False))
    else:
        print(processor.decode(tail.tolist(), skip_special_tokens=False))

    print("\n=== ANSWER ONLY (tail, skip specials) ===\n")
    if tok is not None:
        ans = tok.decode(tail.tolist(), skip_special_tokens=True).strip()
    else:
        ans = processor.decode(tail.tolist(), skip_special_tokens=True).strip()

    print(ans if ans else "[EMPTY OUTPUT AFTER SKIP_SPECIAL_TOKENS]")


if __name__ == "__main__":
    main()
