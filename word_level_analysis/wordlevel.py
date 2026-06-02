# token-level perturbation drops -> word-level drops, medical-NER labels, and the
# stats used in the thesis (AOPC, distribution, Friedman + BH).

import re

import numpy as np
import pandas as pd
from scipy import stats

SPM = "\u2581"                 # SentencePiece marks word starts with this
_PUNCT = re.compile(r"[^\w]+", re.UNICODE)
NER_MODEL = "Clinical-AI-Apollo/Medical-NER"


def _word_initial(s):
    return bool(s) and (s.startswith(SPM) or (s.startswith("<") and s.endswith(">")))


def _is_special(s):
    return bool(re.match(r"^<[^>]*>$", s or ""))


def reconstruct_words(df, tokenizer):
    # merge sub-word tokens into words. a word's drop is its first sub-token's
    # drop; the rest are basically deterministic under teacher forcing.
    ids = sorted(set(int(x) for x in df["token_id"]))
    id2str = dict(zip(ids, tokenizer.convert_ids_to_tokens(ids)))

    df = df.sort_values(["sample_idx", "method", "mask_ratio", "position"],
                        kind="stable")
    out = []
    keys = ["sample_idx", "sample_id", "method", "mask_ratio"]
    for (s_idx, s_id, method, ratio), grp in df.groupby(keys, sort=False):
        tok_ids = grp["token_id"].to_numpy()
        pos = grp["position"].to_numpy()
        drop = grp["prob_drop"].to_numpy()
        op = grp["original_prob"].to_numpy()
        mp = grp["masked_prob"].to_numpy()

        init = np.array([_word_initial(id2str.get(int(t), "")) for t in tok_ids])
        if len(init):
            init[0] = True
        bounds = np.flatnonzero(init).tolist() + [len(tok_ids)]

        w = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            raw = [id2str.get(int(t), "") for t in tok_ids[a:b]]
            if _is_special(raw[0]):
                continue
            merged = "".join(s[1:] if s.startswith(SPM) else s for s in raw)
            clean = _PUNCT.sub("", merged).strip().lower()
            if not clean:                      # pure punctuation -> skip
                continue
            out.append(dict(
                sample_idx=int(s_idx), sample_id=int(s_id), method=str(method),
                mask_ratio=float(ratio), word_idx=w, word_text=merged,
                word_clean=clean, first_token_pos=int(pos[a]),
                n_subtokens=int(b - a), prob_drop=float(drop[a]),
                original_prob=float(op[a]), masked_prob=float(mp[a]),
            ))
            w += 1
    return pd.DataFrame(out)


def build_sample_texts(words):
    # rebuild each caption (one slice is enough, the text is the same across
    # methods/ratios) and keep char spans so NER hits map back to word_idx.
    method = words["method"].iloc[0]
    ratio = float(words["mask_ratio"].min())
    sub = words[(words["method"] == method) & (words["mask_ratio"] == ratio)]
    texts = {}
    for s_idx, grp in sub.groupby("sample_idx", sort=True):
        grp = grp.sort_values("word_idx")
        spans, parts, cur = [], [], 0
        for word, widx in zip(grp["word_clean"], grp["word_idx"]):
            if not word:
                continue
            if parts:
                cur += 1
            spans.append((cur, cur + len(word), widx))
            parts.append(word)
            cur += len(word)
        texts[int(s_idx)] = (" ".join(parts), spans)
    return texts


_NER_PIPE = None


def annotate(words, model_id=NER_MODEL):
    # add ner_class: the entity group with the largest char overlap, else OTHER.
    global _NER_PIPE
    if _NER_PIPE is None:
        from transformers import pipeline
        _NER_PIPE = pipeline("token-classification", model=model_id,
                             aggregation_strategy="simple")
    ner = _NER_PIPE
    texts = build_sample_texts(words)
    label = {}
    for n, (s_idx, (text, spans)) in enumerate(texts.items(), 1):
        for _, _, widx in spans:
            label[(s_idx, widx)] = "OTHER"
        if text.strip():
            ents = ner(text)
            for cs, ce, widx in spans:
                best, best_ov = "OTHER", 0
                for e in ents:
                    g = e.get("entity_group", "")
                    ov = max(0, min(ce, int(e["end"])) - max(cs, int(e["start"])))
                    if g and ov > best_ov:
                        best, best_ov = g, ov
                label[(s_idx, widx)] = best
        if n % 50 == 0 or n == len(texts):
            print(f"  NER {n}/{len(texts)}")
    out = words.copy()
    out["ner_class"] = [label.get((int(s), int(w)), "OTHER")
                        for s, w in zip(out["sample_idx"], out["word_idx"])]
    return out


def bh_fdr(p):
    # Benjamini-Hochberg q-values (NaNs passed through).
    p = np.asarray(list(p), float)
    out = np.full_like(p, np.nan)
    m = ~np.isnan(p)
    if not m.any():
        return out
    pv = p[m]
    order = np.argsort(pv)
    adj = pv[order] * pv.size / (np.arange(pv.size) + 1)
    adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
    q = np.empty(pv.size)
    q[order] = adj
    out[m] = q
    return out


def aopc_table(words, drop_other=True):
    # mean drop per (method, mask_ratio) + AOPC (mean over ratios).
    w = words[words["ner_class"] != "OTHER"] if drop_other else words
    piv = w.groupby(["method", "mask_ratio"])["prob_drop"].mean().unstack()
    piv["AOPC"] = piv.mean(axis=1)
    return piv.round(4)


def ner_class_table(words):
    # mean drop per (ner_class, method), averaged over ratios.
    w = words[words["ner_class"] != "OTHER"]
    return (w.groupby(["ner_class", "method"])["prob_drop"].mean()
            .unstack().round(3))


def distribution_stats(words, mask_ratio=0.1, drop_other=True):
    # median / mean / p95 and the share of the drop carried by the top 10% words.
    w = words[words["mask_ratio"] == mask_ratio]
    if drop_other:
        w = w[w["ner_class"] != "OTHER"]
    rows = []
    for method, g in w.groupby("method"):
        v = np.sort(g["prob_drop"].to_numpy())[::-1]
        pos = v[v > 0].sum()
        top = v[:max(1, int(0.1 * len(v)))].sum()
        rows.append(dict(method=method, n=len(v), mean=v.mean(),
                         median=np.median(v), p95=np.percentile(v, 95),
                         tail_share=(top / pos if pos > 0 else np.nan)))
    return pd.DataFrame(rows).set_index("method").round(4)


def friedman_by_cell(words, methods):
    # Friedman test per (mask_ratio, ner_class) on the paired per-word drops.
    pv = (words.pivot_table(index=["sample_idx", "mask_ratio", "ner_class", "word_idx"],
                            columns="method", values="prob_drop", aggfunc="first")
          .dropna().reset_index())
    rows = []
    for (ratio, cls), g in pv.groupby(["mask_ratio", "ner_class"]):
        if len(g) < 3:
            rows.append(dict(mask_ratio=ratio, ner_class=cls, n=len(g),
                             chi2=np.nan, p_value=np.nan))
            continue
        chi2, p = stats.friedmanchisquare(*[g[m].to_numpy() for m in methods])
        rows.append(dict(mask_ratio=ratio, ner_class=cls, n=len(g),
                         chi2=chi2, p_value=p))
    out = pd.DataFrame(rows)
    out["q_value"] = bh_fdr(out["p_value"])
    return out.sort_values(["ner_class", "mask_ratio"]).reset_index(drop=True)


# --- ablation (Table 2) ---

# plain English stopwords; COCO captions have no medical entities so we filter
# these instead of running the medical NER.
STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but", "nor", "so", "yet", "because", "if", "while", "although",
    "though", "whereas", "as",
    "of", "in", "on", "at", "by", "for", "to", "from", "with", "without", "into",
    "onto", "over", "under", "through", "between", "among", "across", "along",
    "around", "behind", "below", "beside", "beyond", "during", "except", "inside",
    "outside", "near", "off", "since", "toward", "towards", "upon", "via", "within",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must", "ought",
    "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "not", "no", "yes", "there", "here", "then", "than", "too", "also", "very",
    "just", "only", "even", "still", "again",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "one", "two",
    "s", "t", "d", "ll", "ve", "re", "m", "o",
}


def drop_stopwords(words):
    return words[~words["word_clean"].astype(str).isin(STOPWORDS)]


def attention_aopc(words):
    sub = words[words["method"] == "attention"]
    return float(sub.groupby("mask_ratio")["prob_drop"].mean().mean())


def clean_roco(words):
    # ROCOv2: label with medical NER, keep only words that hit an entity.
    words = annotate(words)
    return words[words["ner_class"] != "OTHER"]
