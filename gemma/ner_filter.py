"""
NER-based output filtering for the saliency evaluation pipeline.

Uses a Medical-NER model to classify tokens in the generated text,
then produces separate "variants" of the output – one per entity group –
containing only the tokens belonging to that group.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import pipeline as hf_pipeline

from config import Config


_ner_pipe = None  # lazy singleton


def get_ner_pipeline(cfg: Config):
    """Return (and cache) the NER pipeline."""
    global _ner_pipe
    if _ner_pipe is None:
        print(f"[NER] Loading {cfg.ner_model} …")
        _ner_pipe = hf_pipeline(
            "token-classification",
            model=cfg.ner_model,
            aggregation_strategy="simple",
        )
        print("[NER] Ready.")
    return _ner_pipe


def run_ner(text: str, cfg: Config) -> List[dict]:
    """Run NER on *text* and return the raw entity list."""
    pipe = get_ner_pipeline(cfg)
    return pipe(text)


def build_variants(
    generated_text: str,
    generated_ids: torch.Tensor,
    input_len: int,
    tokenizer,
    cfg: Config,
) -> Dict[str, dict]:
    """Build output variants based on NER entity groups.

    Returns a dict mapping variant name → info dict::

        {
            "original": {
                "label": "original",
                "token_positions": [pos0, pos1, ...],  # all generated positions
                "text": "full generated text",
            },
            "DISEASE_DISORDER": {
                "label": "DISEASE_DISORDER",
                "token_positions": [pos3, pos7, ...],   # only matching positions
                "text": "extracted span text",
            },
            ...
        }

    Entity groups that do not appear in the output are silently skipped.
    """
    total_len = generated_ids.shape[1]
    all_gen_positions = list(range(input_len, total_len))

    variants: Dict[str, dict] = {
        "original": {
            "label": "original",
            "token_positions": all_gen_positions,
            "text": generated_text,
        }
    }

    if not cfg.use_ner_filter:
        return variants

    entities = run_ner(generated_text, cfg)
    if not entities:
        print("  [NER] No entities found – only 'original' variant.")
        return variants

    # Group entities by entity_group
    groups: Dict[str, List[dict]] = {}
    for ent in entities:
        grp = ent["entity_group"]
        if grp in cfg.ner_entity_groups:
            groups.setdefault(grp, []).append(ent)

    # For each group, figure out which token positions correspond to the
    # entity spans.  We do this by mapping character offsets in
    # generated_text back to token positions.
    #
    # Build a map: char_offset → token_position
    token_char_spans: List[Tuple[int, int, int]] = []  # (start, end, position)
    char_cursor = 0
    for pos in all_gen_positions:
        tid = generated_ids[0, pos].item()
        tok_str = tokenizer.decode([tid], skip_special_tokens=True)
        # Find this token string in the remaining generated text
        # (character-level alignment)
        idx = generated_text.find(tok_str, char_cursor) if tok_str else -1
        if idx >= 0:
            token_char_spans.append((idx, idx + len(tok_str), pos))
            char_cursor = idx + len(tok_str)
        else:
            # Fallback: assign current cursor position
            token_char_spans.append((char_cursor, char_cursor, pos))

    for grp, ent_list in groups.items():
        matched_positions = []
        matched_texts = []
        for ent in ent_list:
            ent_start = ent["start"]
            ent_end = ent["end"]
            matched_texts.append(ent["word"].strip())
            for tok_start, tok_end, pos in token_char_spans:
                # Token overlaps with entity span
                if tok_end > ent_start and tok_start < ent_end:
                    if pos not in matched_positions:
                        matched_positions.append(pos)

        if matched_positions:
            matched_positions.sort()
            variants[grp] = {
                "label": grp,
                "token_positions": matched_positions,
                "text": " | ".join(matched_texts),
            }
            print(f"  [NER] {grp}: {len(matched_positions)} tokens – "
                  f"{variants[grp]['text'][:80]}")

    if len(variants) == 1:
        print("  [NER] No configured entity groups found in output.")

    return variants
