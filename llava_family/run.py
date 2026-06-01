#!/usr/bin/env python3
"""
Main pipeline for: *Evaluating Explainability Methods for Medical
Vision-Language Models via Perturbation-Based Faithfulness Testing*.

LLaVA-Med variant.

Usage examples
--------------
# Quick smoke test (2 samples, attention only, average perturbation):
  python run.py --num-samples 2 --methods attention --eval-mode average

# Full experiment (50 samples, all methods, per-token perturbation):
    python run.py --num-samples 50 --methods attention gradcam gmar_l1 gmar_l2 \
                --eval-mode per_token

# Using a local image folder instead of HuggingFace dataset:
  python run.py --dataset ./my_images --num-samples 10

# Disable 4-bit quantisation (needs ≥ 16 GB VRAM):
  python run.py --no-4bit --num-samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Local imports
from config import Config
from dataset import load_dataset_samples
from model_utils import (
    load_model_and_processor,
    generate_caption,
    get_tokenizer,
    get_image_token_positions,
    get_token_probabilities,
    get_content_token_mask,
    build_tf_inputs,
    move_inputs_to_device,
)
from saliency import get_saliency_fn
from evaluation import (
    evaluate_faithfulness_average,
    evaluate_faithfulness_per_token,
    evaluate_faithfulness_random,
)
from visualization import (
    save_token_saliency_grid,
    save_comparison_figure,
    save_perturbation_curve,
    save_aggregate_curves,
)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="VLM Explainability – perturbation-based evaluation (LLaVA-Med)"
    )
    ap.add_argument("--model", default=Config.model_id,
                    help="HuggingFace model id")
    ap.add_argument("--dataset", default=Config.dataset_name,
                    help="HuggingFace dataset id or local path")
    ap.add_argument("--dataset-split", default=Config.dataset_split)
    ap.add_argument("--image-column", default=Config.image_column)
    ap.add_argument("--caption-column", default=Config.caption_column)
    ap.add_argument("--num-samples", type=int, default=Config.num_samples)
    ap.add_argument("--max-new-tokens", type=int, default=Config.max_new_tokens)
    ap.add_argument("--prompt", default=Config.prompt)
    ap.add_argument("--methods", nargs="+",
                    default=["attention", "gradcam", "gmar_l1", "gmar_l2"],
                    choices=["attention", "gradcam", "gmar_l1", "gmar_l2"])
    ap.add_argument("--eval-mode", default="average",
                    choices=["average", "per_token"],
                    help="'average' = fast (avg saliency); "
                         "'per_token' = detailed (slow)")
    ap.add_argument("--content-only", action="store_true", default=True,
                    help="Evaluate only content tokens in per_token mode")
    ap.add_argument("--no-content-only", dest="content_only",
                    action="store_false")
    ap.add_argument("--mask-ratios", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--no-4bit", action="store_true",
                    help="Load in float16 instead of 4-bit")
    ap.add_argument("--attn-impl", default="eager",
                    choices=["eager", "sdpa"],
                    help="Attention implementation (eager needed for "
                         "attention saliency)")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--no-vis", action="store_true",
                    help="Skip saving visualisation images")
    ap.add_argument("--attention-layers", default="global",
                    help="Layer aggregation for attention saliency: "
                         "global | all | lastN")
    args = ap.parse_args()

    cfg = Config()
    cfg.model_id = args.model
    cfg.dataset_name = args.dataset
    cfg.dataset_split = args.dataset_split
    cfg.image_column = args.image_column
    cfg.caption_column = args.caption_column
    cfg.num_samples = args.num_samples
    cfg.max_new_tokens = args.max_new_tokens
    cfg.prompt = args.prompt
    cfg.methods = args.methods
    cfg.mask_ratios = args.mask_ratios
    cfg.load_in_4bit = not args.no_4bit
    cfg.attn_implementation = args.attn_impl
    cfg.output_dir = args.output_dir
    cfg.save_visualizations = not args.no_vis
    cfg.attention_layer_strategy = args.attention_layers
    return cfg, args.eval_mode, args.content_only


# ──────────────────────────────────────────────────────────────────────────
# Per-sample processing
# ──────────────────────────────────────────────────────────────────────────
def process_sample(
    model,
    processor,
    sample: dict,
    sample_idx: int,
    cfg: Config,
    eval_mode: str,
    content_only: bool,
    out_dir: Path,
) -> dict:
    """Run saliency extraction + perturbation evaluation on one sample."""

    image = sample["image"]
    ref_caption = sample.get("caption", "")
    sample_id = sample.get("id", str(sample_idx))
    tok = get_tokenizer(processor)

    # ── 1. Generate caption ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Sample {sample_idx}: {sample_id}")
    print(f"{'='*60}")
    gen_ids, gen_text, input_len, inputs = generate_caption(
        model, processor, image, cfg
    )
    print(f"  Generated ({gen_ids.shape[1] - input_len} tokens): {gen_text[:120]}")
    if ref_caption:
        print(f"  Reference: {ref_caption[:120]}")

    total_len = gen_ids.shape[1]
    num_gen = total_len - input_len
    if num_gen == 0:
        print("  ⚠ Model generated 0 tokens – skipping.")
        return {}

    # ── 2. Image-token positions ─────────────────────────────────────────
    img_positions = get_image_token_positions(inputs, cfg.image_token_index)
    print(f"  Image tokens: {len(img_positions)}")

    # ── 3. Build teacher-forcing inputs ──────────────────────────────────
    tf_inputs = build_tf_inputs(inputs, gen_ids, input_len)

    # ── 4. Original token probabilities ──────────────────────────────────
    orig_probs = get_token_probabilities(model, tf_inputs, gen_ids, input_len)
    print(f"  Mean original prob: {orig_probs.mean():.4f}")

    # ── 5. Token strings (for visualisation / logging) ───────────────────
    token_strings: dict[int, str] = {}
    for pos in range(input_len, total_len):
        tid = gen_ids[0, pos].item()
        token_strings[pos] = tok.decode([tid], skip_special_tokens=True).strip()

    content_mask = get_content_token_mask(tok, gen_ids, input_len)
    n_content = sum(content_mask)
    print(f"  Content tokens: {n_content}/{num_gen}")

    # ── 6. Saliency + evaluation per method ─────────────────────────────
    sample_dir = out_dir / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    all_saliency: dict[str, dict[int, np.ndarray]] = {}
    method_eval_results: dict[str, dict] = {}
    random_eval_result: dict | None = None
    content_positions: list[int] = [
        pos for pos, keep in zip(
            range(input_len, total_len), content_mask
        ) if keep
    ]

    for method in cfg.methods:
        print(f"\n  ── {method} ──")
        compute = get_saliency_fn(method)

        t0 = time.time()
        sal_maps = compute(
            model, tf_inputs, gen_ids, input_len, img_positions, cfg
        )
        dt = time.time() - t0
        print(f"  Saliency computed in {dt:.1f}s  ({len(sal_maps)} maps)")
        all_saliency[method] = sal_maps

        eval_content_mask = [
            (content_mask[pos - input_len] if content_only else True)
            for pos in range(input_len, total_len)
        ]

        t0 = time.time()
        if eval_mode == "average":
            ev = evaluate_faithfulness_average(
                model, inputs, gen_ids, input_len,
                sal_maps, orig_probs, cfg,
            )
        else:
            ev = evaluate_faithfulness_per_token(
                model, inputs, gen_ids, input_len,
                sal_maps, orig_probs, cfg,
                content_mask=eval_content_mask,
            )
        dt = time.time() - t0
        print(f"  Perturbation eval in {dt:.1f}s   AOPC = {ev['aopc']:.4f}")
        for mr, drop in ev["mean_drops_by_ratio"].items():
            print(f"    mask {mr:.0%}: Δp = {drop:+.4f}")
        method_eval_results[method] = ev

        # Visualisations (full saliency, not per-variant)
        if cfg.save_visualizations:
            save_token_saliency_grid(
                image, sal_maps, token_strings, method,
                str(sample_dir / f"saliency_{method}.png"),
                content_positions=content_positions,
            )

    # ── 7. Random baseline ───────────────────────────────────────────────
    print("\n  ── random baseline ──")
    t0 = time.time()
    random_positions = list(range(input_len, total_len))
    random_eval_result = evaluate_faithfulness_random(
        model, inputs, gen_ids, input_len, orig_probs, cfg,
        content_mask=content_mask if content_only else None,
        token_positions=random_positions,
    )
    dt = time.time() - t0
    print(f"  Random eval in {dt:.1f}s   AOPC = {random_eval_result['aopc']:.4f}")
    method_eval_results["random"] = random_eval_result

    # ── 8. Comparison figure ─────────────────────────────────────────────
    if cfg.save_visualizations and len(all_saliency) >= 2:
        save_comparison_figure(
            image, all_saliency, token_strings,
            str(sample_dir / "comparison.png"),
            content_positions=content_positions,
        )

    # ── 9. Per-sample perturbation curve ─────────────────────────────────
    if cfg.save_visualizations:
        save_perturbation_curve(
            method_eval_results,
            str(sample_dir / "perturbation_curve.png"),
            title=f"Sample {sample_idx}",
        )

    # ── 10. Save original image for reference ────────────────────────────
    image.save(str(sample_dir / "original.png"))

    # ── 11. Cleanup GPU memory ───────────────────────────────────────────
    del tf_inputs, orig_probs, sal_maps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sample_id": sample_id,
        "sample_idx": sample_idx,
        "generated_text": gen_text,
        "reference_caption": ref_caption,
        "num_generated_tokens": num_gen,
        "num_content_tokens": n_content,
        "eval": {m: {
            "aopc": r["aopc"],
            "mean_drops_by_ratio": r["mean_drops_by_ratio"],
        } for m, r in method_eval_results.items()},
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    cfg, eval_mode, content_only = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── save config ──────────────────────────────────────────────────────
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(cfg), f, indent=2, default=str)

    # ── load model ───────────────────────────────────────────────────────
    model, processor = load_model_and_processor(cfg)

    # ── load dataset ─────────────────────────────────────────────────────
    samples = load_dataset_samples(cfg)

    # ── process each sample ──────────────────────────────────────────────
    all_sample_results: list[dict] = []
    all_eval_by_method: dict[str, list[dict]] = {}

    t_total = time.time()
    for i, sample in enumerate(samples):
        res = process_sample(
            model, processor, sample, i, cfg,
            eval_mode, content_only, out_dir,
        )
        if not res:
            continue
        all_sample_results.append(res)

        for key in res.get("eval", {}):
            all_eval_by_method.setdefault(key, []).append(res["eval"][key])

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All {len(all_sample_results)} samples processed in {elapsed:.0f}s")
    print(f"{'='*60}\n")

    # ── aggregate results ────────────────────────────────────────────────
    summary_rows = []
    for method, evals in all_eval_by_method.items():
        if not evals:
            continue
        aopc_vals = [e["aopc"] for e in evals]
        mean_aopc = float(np.mean(aopc_vals))
        std_aopc = float(np.std(aopc_vals))
        print(f"  {method:>12s}:  AOPC = {mean_aopc:.4f} ± {std_aopc:.4f}")

        summary_rows.append({
            "method": method,
            "mean_aopc": mean_aopc,
            "std_aopc": std_aopc,
            "n_samples": len(evals),
        })

        # Per-mask-ratio aggregation
        all_ratios = set()
        for e in evals:
            all_ratios.update(e.get("mean_drops_by_ratio", {}).keys())
        for r in sorted(all_ratios):
            drops = [
                e["mean_drops_by_ratio"][r]
                for e in evals
                if r in e.get("mean_drops_by_ratio", {})
            ]
            print(f"    mask {r:.0%}: Δp = {np.mean(drops):+.4f} "
                  f"± {np.std(drops):.4f}")

    # ── save CSV summary ─────────────────────────────────────────────────
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = out_dir / "summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSummary saved to {csv_path}")

    # ── save detailed JSON ───────────────────────────────────────────────
    json_path = out_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_sample_results, f, indent=2, default=str)
    print(f"Detailed results saved to {json_path}")

    # ── aggregate perturbation curve ─────────────────────────────────────
    if cfg.save_visualizations and any(all_eval_by_method.values()):
        save_aggregate_curves(
            all_eval_by_method,
            str(out_dir / "aggregate_perturbation.png"),
        )
        print(f"Aggregate perturbation plot saved to {out_dir / 'aggregate_perturbation.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
