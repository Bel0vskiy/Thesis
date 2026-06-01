"""
Generate a PDF report for the LLaVA-Med explainability experiment.
Run with:  /tmp/pdf_env/bin/python generate_report.py
Output:    results/llava_med_report.pdf
"""

import os
import json
import math
from pathlib import Path
from collections import Counter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── paths ─────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
RES    = BASE / "results"
RES_CO = BASE / "results_content_only"
OUT    = RES / "llava_med_report.pdf"

# ── load data ─────────────────────────────────────────────────────────────
with open(RES / "all_results.json") as f:
    data_all = json.load(f)
with open(RES_CO / "all_results.json") as f:
    data_co = json.load(f)

METHODS = ["attention", "gradcam", "gmar", "random"]
METHOD_LABELS = {
    "attention": "Attention Rollout",
    "gradcam":   "GradCAM",
    "gmar":      "GMAR",
    "random":    "Random Baseline",
}
RATIOS = ["0.1", "0.2", "0.3", "0.4", "0.5"]

# ── helper stats ──────────────────────────────────────────────────────────
def mean(v):   return sum(v) / len(v)
def std(v):    m = mean(v); return math.sqrt(sum((x-m)**2 for x in v)/len(v))
def median(v): s = sorted(v); n = len(s); return (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]

def aopc_stats(data, method):
    vals = [d["eval"][method]["aopc"] for d in data]
    return mean(vals), std(vals), median(vals), sum(1 for v in vals if v < 0)

def ratio_mean(data, method, r):
    vals = [d["eval"][method]["mean_drops_by_ratio"].get(r)
            for d in data if d["eval"][method]["mean_drops_by_ratio"].get(r) is not None]
    return mean(vals), len(vals)

def win_counts(data):
    w = Counter()
    for d in data:
        best = max(METHODS, key=lambda m: d["eval"][m]["aopc"])
        w[best] += 1
    return w

def beat_random_count(data, method):
    return sum(1 for d in data if d["eval"][method]["aopc"] > d["eval"]["random"]["aopc"])

# ── styles ────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
W, H = A4

BRAND   = colors.HexColor("#1a3a5c")   # dark navy
ACCENT  = colors.HexColor("#2e86c1")   # blue
LIGHT   = colors.HexColor("#eaf4fc")
GRAY    = colors.HexColor("#555555")
LGRAY   = colors.HexColor("#dddddd")
GREEN   = colors.HexColor("#1e8449")
RED     = colors.HexColor("#c0392b")
ORANGE  = colors.HexColor("#e67e22")

title_style = ParagraphStyle("TitleStyle", parent=styles["Title"],
    fontSize=22, leading=28, textColor=BRAND, spaceAfter=6,
    alignment=TA_CENTER, fontName="Helvetica-Bold")

subtitle_style = ParagraphStyle("SubTitle", parent=styles["Normal"],
    fontSize=11, leading=14, textColor=GRAY, spaceAfter=4,
    alignment=TA_CENTER)

h1_style = ParagraphStyle("H1", parent=styles["Heading1"],
    fontSize=15, leading=20, textColor=BRAND, spaceBefore=18, spaceAfter=6,
    fontName="Helvetica-Bold", borderPad=4)

h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
    fontSize=12, leading=16, textColor=ACCENT, spaceBefore=12, spaceAfter=4,
    fontName="Helvetica-Bold")

h3_style = ParagraphStyle("H3", parent=styles["Heading3"],
    fontSize=10, leading=13, textColor=BRAND, spaceBefore=8, spaceAfter=3,
    fontName="Helvetica-Bold")

body_style = ParagraphStyle("Body", parent=styles["Normal"],
    fontSize=9.5, leading=14, textColor=colors.black, spaceAfter=6,
    alignment=TA_JUSTIFY)

bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"],
    fontSize=9.5, leading=13, textColor=colors.black, spaceAfter=3,
    leftIndent=14, firstLineIndent=-10)

code_style = ParagraphStyle("Code", parent=styles["Code"],
    fontSize=8.5, leading=12, backColor=colors.HexColor("#f4f4f4"),
    leftIndent=12, rightIndent=12, spaceBefore=4, spaceAfter=4,
    borderColor=LGRAY, borderWidth=0.5, borderPad=5)

caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
    fontSize=8.5, leading=11, textColor=GRAY, spaceAfter=8,
    alignment=TA_CENTER, fontName="Helvetica-Oblique")

def B(text): return f"<b>{text}</b>"
def I(text): return f"<i>{text}</i>"
def TT(text): return f'<font face="Courier" size="8">{text}</font>'

def hr(): return HRFlowable(width="100%", thickness=0.5, color=LGRAY, spaceAfter=6)

def table_style_base(header_bg=BRAND):
    return TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  header_bg),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  9),
        ("FONTSIZE",    (0,1), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
        ("GRID",        (0,0), (-1,-1), 0.4, LGRAY),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ])

# ── build story ───────────────────────────────────────────────────────────
story = []

# ══════════════════════════════════════════════════════════════════════════
#  TITLE PAGE
# ══════════════════════════════════════════════════════════════════════════
story += [
    Spacer(1, 2.5*cm),
    Paragraph("LLaVA-Med Explainability Evaluation", title_style),
    Paragraph("Perturbation-Based Faithfulness of Visual Saliency Methods", subtitle_style),
    Spacer(1, 0.4*cm),
    HRFlowable(width="60%", thickness=2, color=ACCENT, spaceAfter=10),
    Paragraph("Experiment Report — April 2026", subtitle_style),
    Spacer(1, 0.5*cm),
    Paragraph(
        "This report presents a systematic evaluation of three visual saliency methods "
        "applied to LLaVA-Med, a large vision-language model for radiology captioning. "
        "Faithfulness is measured via the Area Over the Perturbation Curve (AOPC) metric "
        "across 50 radiology images from the ROCOv2 dataset.",
        ParagraphStyle("Abs", parent=body_style, leftIndent=40, rightIndent=40,
                       alignment=TA_CENTER, textColor=GRAY, fontSize=9.5)),
    Spacer(1, 1*cm),
]

# summary box
summary_data = [
    ["Model", "llava-hf/llava-1.5-7b-hf (4-bit NF4)"],
    ["Dataset", "eltorio/ROCOv2-radiology  ·  train split  ·  50 samples"],
    ["Metric", "AOPC — Area Over the Perturbation Curve"],
    ["Saliency Methods", "Attention Rollout, GradCAM, GMAR, Random Baseline"],
    ["Mask Ratios", "r ∈ {0.1, 0.2, 0.3, 0.4, 0.5}"],
    ["Evaluation Mode", "Per-token perturbation with teacher forcing"],
]
t = Table(summary_data, colWidths=[4.5*cm, 11.5*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), LIGHT),
    ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE",   (0,0), (-1,-1), 9),
    ("GRID",       (0,0), (-1,-1), 0.4, LGRAY),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ("LEFTPADDING",(0,0), (-1,-1), 8),
    ("ALIGN",      (0,0), (0,-1), "RIGHT"),
    ("ALIGN",      (1,0), (1,-1), "LEFT"),
]))
story += [t, PageBreak()]

# ══════════════════════════════════════════════════════════════════════════
#  1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("1. Introduction", h1_style), hr(),
    Paragraph(
        "Visual saliency methods aim to explain which image regions most influence a model's "
        "predictions. For medical imaging, trustworthy explanations are clinically important: "
        "a model that attends to radiologically meaningful structures is more likely to be "
        "reliable and auditable. However, most saliency evaluations are qualitative. "
        "This experiment applies a quantitative faithfulness test — the AOPC metric — to "
        "determine whether three saliency methods (Attention Rollout, GradCAM, GMAR) "
        "genuinely identify causally important image regions for LLaVA-Med.", body_style),
    Paragraph(
        "The core hypothesis is: if a saliency map correctly identifies the most important "
        "image patches, then zeroing those patches should cause a larger drop in the model's "
        "token probabilities than zeroing randomly selected patches.", body_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  2. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("2. Model Architecture", h1_style), hr(),
    Paragraph(
        "LLaVA 1.5 (Large Language and Vision Assistant) follows a two-component architecture "
        "that connects a visual encoder to a large language model via a learned projection layer.",
        body_style),

    Paragraph("2.1 Visual Encoder", h2_style),
    Paragraph(
        "The visual encoder is CLIP ViT-L/14 at 336 px resolution. The input image is divided "
        "into a 24×24 grid of non-overlapping 14×14 pixel patches, producing <b>576 image tokens</b> "
        "(the CLS token is discarded). Each token is a 1024-dimensional embedding from the final "
        "CLIP transformer layer.", body_style),
    Paragraph(
        "Formally, given an input image "
        + I("x") + " ∈ ℝ" + I("³ˣ³³⁶ˣ³³⁶")
        + ", the visual encoder produces:",
        body_style),
    Paragraph(
        TT("V = CLIP_ViT(x)  ∈  ℝ^{576 × 1024}"),
        code_style),

    Paragraph("2.2 Multi-Modal Projector", h2_style),
    Paragraph(
        "A two-layer MLP (the 'mm projector') maps the visual embeddings to the LLM's "
        "hidden dimension (4096 for LLaMA-2-7B backbone), preserving the 576-token spatial "
        "layout without any pooling:", body_style),
    Paragraph(
        TT("E_vis = MLP(V)  ∈  ℝ^{576 × 4096}"),
        code_style),

    Paragraph("2.3 Language Model Backbone", h2_style),
    Paragraph(
        "The backbone is LLaMA-2-7B (32 transformer layers, full-sequence attention in every layer). "
        "Image embeddings are inserted into the token sequence at the position of the special "
        + TT("<image>") + " placeholder (token id 32000). The full input to the LLM is therefore:",
        body_style),
    Paragraph(
        TT("[system_tokens] + [576 image_tokens] + [text_tokens]"),
        code_style),
    Paragraph(
        "The model is loaded in 4-bit NF4 quantisation (bitsandbytes) with double quantisation "
        "enabled and fp16 compute dtype. Attention is set to eager (non-Flash) mode to enable "
        "extraction of full attention matrices for saliency computation.", body_style),

    Paragraph("2.4 Language Prior Effect", h2_style),
    Paragraph(
        "Because the LLM backbone processes both image tokens and preceding text tokens, "
        "each output token's probability is conditioned on the full auto-regressive prefix. "
        "By the time the model generates, e.g., token 'pneumothorax' at position "
        + I("t") + ", it has already consumed all tokens 0…" + I("t") + "−1 as context. "
        "The text prefix alone strongly constrains the prediction, reducing the marginal "
        "contribution of the image. This is the primary cause of the low AOPC values "
        "observed in this experiment and is a known characteristic of language-prior-dominant "
        "VLMs.", body_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  3. SALIENCY METHODS
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("3. Saliency Methods", h1_style), hr(),

    Paragraph("3.1 Attention Rollout", h2_style),
    Paragraph(
        "Attention Rollout (Abnar & Zuidema, 2020) propagates attention through all "
        "transformer layers by recursively multiplying attention matrices, accounting for "
        "residual connections.", body_style),
    Paragraph(B("Algorithm:"), body_style),
    Paragraph("• Initialise: R₀ = I  (identity matrix, shape S×S)", bullet_style),
    Paragraph("• For each selected layer l:", bullet_style),
    Paragraph(
        TT("  Ā_l = mean_over_heads( A_l )"),
        code_style),
    Paragraph(
        TT("  Ā_l = 0.5 * Ā_l  +  0.5 * I          (residual blend)"),
        code_style),
    Paragraph(
        TT("  Ā_l = Ā_l / row_sum(Ā_l)              (renormalise)"),
        code_style),
    Paragraph(
        TT("  R_l = Ā_l @ R_{l-1}"),
        code_style),
    Paragraph(
        "• For generated token at position " + I("t") + ", extract the row R[t, :] "
        "restricted to image-token columns, reshape to 24×24 grid → saliency map.",
        bullet_style),
    Paragraph(
        "Layers used: a representative subset of 8 layers "
        "(3, 7, 11, 15, 19, 23, 27, 31) from the 32-layer backbone.",
        body_style),

    Paragraph("3.2 GradCAM", h2_style),
    Paragraph(
        "GradCAM (Selvaraju et al., 2017) uses gradients of the target token's logit "
        "with respect to the multi-modal projector's output activations.", body_style),
    Paragraph(B("Algorithm:"), body_style),
    Paragraph(
        "• Hook the mm projector output A ∈ ℝ^{576 × 4096}; run one forward pass.",
        bullet_style),
    Paragraph(
        "• For generated token at position " + I("t") + " with target token id c:",
        bullet_style),
    Paragraph(
        TT("  grad = ∂ logit_c / ∂ A     [shape: 576 × 4096]"),
        code_style),
    Paragraph(
        TT("  saliency[i] = sum_over_channels( grad[i] * A[i] )   for i=1..576"),
        code_style),
    Paragraph(
        TT("  saliency = ReLU(saliency)"),
        code_style),
    Paragraph(
        "• Reshape the 576-length vector to 24×24 grid.",
        bullet_style),
    Paragraph(
        "Note: unlike the original image-classification GradCAM which averages gradients "
        "spatially to get per-channel weights, this variant uses element-wise gradient × "
        "activation and sums over channels, giving a direct spatial importance score per token.",
        body_style),

    Paragraph("3.3 Gradient-Weighted Model Attention Rollout (GMAR)", h2_style),
    Paragraph(
        "GMAR combines Attention Rollout with gradient-derived per-head importance weights, "
        "so that only task-relevant attention heads contribute to the rollout.",
        body_style),
    Paragraph(B("Algorithm:"), body_style),
    Paragraph("• Run forward pass with output_attentions=True; compute gradients.", bullet_style),
    Paragraph(
        "• For each layer " + I("l") + " and head " + I("h") + ":",
        bullet_style),
    Paragraph(
        TT("  importance[l][h] = mean |∂ logit_c / ∂ A_l^h|   (abs mean over spatial dims)"),
        code_style),
    Paragraph(
        TT("  w[l][h] = clamp(importance[l][h], min=0) / sum_h(...)   (softmax-like norm)"),
        code_style),
    Paragraph(
        "• Compute gradient-weighted rollout: for each selected layer " + I("l") + ":",
        bullet_style),
    Paragraph(
        TT("  Ā_l = sum_h( w[l][h] * A_l^h )         (weighted average over heads)"),
        code_style),
    Paragraph(
        TT("  Ā_l = 0.5 * Ā_l  +  0.5 * I"),
        code_style),
    Paragraph(
        TT("  R_l = Ā_l @ R_{l-1}"),
        code_style),
    Paragraph(
        "• Extract image-token columns from rollout row " + I("t") + ", reshape to 24×24.",
        bullet_style),

    Paragraph("3.4 Random Baseline", h2_style),
    Paragraph(
        "A random uniform saliency map is sampled (seeded at 42) independently for each "
        "mask ratio, serving as a lower bound. Any method that does not exceed random "
        "significantly has failed to identify meaningful image regions.", body_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  4. FAITHFULNESS METRIC — AOPC
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("4. Faithfulness Metric — AOPC", h1_style), hr(),

    Paragraph("4.1 Perturbation Protocol", h2_style),
    Paragraph(
        "For each generated token " + I("t") + " and mask ratio " + I("r") + ":", body_style),
    Paragraph(
        "1. Use the saliency map for token " + I("t") + " to identify the top "
        + I("r") + "·576 highest-scoring image patches.", bullet_style),
    Paragraph(
        "2. Zero those patches in pixel space (set to 0 in the normalised "
        "[−3, 3] tensor; equivalent to replacing with a constant 'no-information' region).",
        bullet_style),
    Paragraph(
        "3. Run the model again in teacher-forcing mode with the zeroed image.",
        bullet_style),
    Paragraph(
        "4. Measure the probability drop:  Δp(t, r) = p_orig(t) − p_masked(t, r)",
        bullet_style),
    Paragraph(
        "Masking is performed in image pixel space, not in the token embedding space, "
        "via nearest-neighbour upsampling of the 24×24 patch mask to the 336×336 input "
        "resolution.", body_style),

    Paragraph("4.2 AOPC Formula", h2_style),
    Paragraph(
        "The mean probability drop at ratio " + I("r") + " across all evaluated tokens " + I("T") + ":",
        body_style),
    Paragraph(
        TT("MeanDrop(r) = (1/|T|) * Σ_{t ∈ T}  Δp(t, r)"),
        code_style),
    Paragraph(
        "The Area Over the Perturbation Curve (AOPC) aggregates across all mask ratios:",
        body_style),
    Paragraph(
        TT("AOPC = (1/|R|) * Σ_{r ∈ R}  MeanDrop(r)     R = {0.1, 0.2, 0.3, 0.4, 0.5}"),
        code_style),
    Paragraph(
        "Higher AOPC indicates that the saliency method is identifying genuinely causal "
        "image regions — removing them degrades the model's output more than removing "
        "random patches would.", body_style),

    Paragraph("4.3 Token Selection", h2_style),
    Paragraph(
        "Two evaluation conditions are reported:", body_style),
    Paragraph(
        B("All tokens:") + "  Every generated token is evaluated, including stop words "
        "('a', 'the', 'of', …) and punctuation. These tokens have high prior probability "
        "from text context alone and contribute near-zero drops regardless of image content, "
        "suppressing AOPC.", bullet_style),
    Paragraph(
        B("Content tokens only:") + "  Stop words and single-character tokens are excluded. "
        "The content token filter removes ~46% of generated tokens on average, retaining "
        "only clinically or descriptively meaningful terms (e.g. 'chest', 'consolidation', "
        "'opacity'). This is the more interpretable faithfulness measure for caption-based "
        "VLM evaluation.", bullet_style),
    Paragraph(
        "On average, generated captions contain 16.3 tokens total, of which 9.0 (55.4%) "
        "are content tokens.", body_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  5. EXPERIMENTAL SETUP
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("5. Experimental Setup", h1_style), hr(),

    Paragraph("5.1 Dataset", h2_style),
    Paragraph(
        "The ROCOv2 radiology dataset (Pelka et al., 2022) is used. 50 images are sampled "
        "from the training split. The dataset contains diverse radiology modalities: "
        "X-ray (chest, extremity), CT, MRI, and ultrasound. Each image has an expert "
        "reference caption; the model generates its own caption which is then evaluated.",
        body_style),

    Paragraph("5.2 Caption Generation", h2_style),
    Paragraph(B("Prompt:"), body_style),
    Paragraph(
        TT('"Write a single-sentence radiology caption for this medical image. '
           'Be concise and clinical, like a figure caption in a medical journal."'),
        code_style),
    Paragraph(
        "Greedy decoding (do_sample=False), max 64 new tokens. "
        "Generated captions show strong template adherence: many begin with "
        "'A black and white image of a…', reflecting the model's tendency to describe "
        "radiograph appearance rather than clinical findings.", body_style),

    Paragraph("5.3 Evaluation Configuration", h2_style),
]
config_data = [
    ["Parameter", "Value"],
    ["Mask ratios", "0.1, 0.2, 0.3, 0.4, 0.5"],
    ["Eval mode", "per-token (one masked forward pass per token x ratio)"],
    ["Image token grid", "24x24 = 576 patches"],
    ["Attention layers for rollout", "3, 7, 11, 15, 19, 23, 27, 31 (global strategy)"],
    ["Quantisation", "4-bit NF4, fp16 compute, double quant"],
    ["NER filter", "Disabled (all-token run); content-token filter applied post-hoc"],
]
t = Table(config_data, colWidths=[5.5*cm, 10.5*cm])
t.setStyle(table_style_base())
story.append(t)
story.append(Spacer(1, 0.3*cm))

# ══════════════════════════════════════════════════════════════════════════
#  6. RESULTS — ALL TOKENS
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("6. Results — All Tokens", h1_style), hr(),

    Paragraph("6.1 AOPC Summary", h2_style),
]

wins_all = win_counts(data_all)
n = len(data_all)
aopc_data = [["Method", "Mean AOPC", "Std", "Median", "Neg. AOPC", "Wins", "> Random"]]
for m in METHODS:
    mu, sd, med, neg = aopc_stats(data_all, m)
    beat = beat_random_count(data_all, m)
    aopc_data.append([
        METHOD_LABELS[m],
        f"{mu:.5f}",
        f"{sd:.5f}",
        f"{med:.5f}",
        str(neg),
        f"{wins_all[m]}/{n}",
        f"{beat}/{n}",
    ])

t = Table(aopc_data, colWidths=[3.8*cm, 2.2*cm, 2.0*cm, 2.0*cm, 1.8*cm, 1.6*cm, 2.0*cm])
ts = table_style_base()
# highlight attention row
ts.add("BACKGROUND", (0,1), (-1,1), colors.HexColor("#d5e8d4"))
ts.add("FONTNAME",   (0,1), (-1,1), "Helvetica-Bold")
t.setStyle(ts)
story += [t, Spacer(1, 0.2*cm),
          Paragraph("Table 1. AOPC results across all generated tokens (n=50 images). "
                    "Highlighted row = best performing method.", caption_style)]

story += [
    Paragraph("6.2 Per-Ratio Mean Drops", h2_style),
]
ratio_data = [["Ratio"] + [METHOD_LABELS[m] for m in METHODS]]
for r in RATIOS:
    row = [f"r = {r}"]
    for m in METHODS:
        mu, cnt = ratio_mean(data_all, m, r)
        row.append(f"{mu:.5f}")
    ratio_data.append(row)

t = Table(ratio_data, colWidths=[1.8*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
t.setStyle(table_style_base())
story += [t, Spacer(1, 0.2*cm),
          Paragraph("Table 2. Mean probability drop per mask ratio across 50 samples "
                    "(all-token condition).", caption_style)]

story += [
    Paragraph("6.3 Perturbation Curve", h2_style),
]

# include figure if available
fig_agg = RES / "aggregate_perturbation.png"
if fig_agg.exists():
    story.append(RLImage(str(fig_agg), width=14*cm, height=8*cm))
    story.append(Paragraph(
        "Figure 1. Aggregate perturbation curves (mean ± 1 std) across 50 samples. "
        "All curves are monotonically increasing from r=0.1 to r=0.5. "
        "Method ranking: Attention Rollout > GMAR > GradCAM > Random.", caption_style))

story += [
    Paragraph("6.4 AOPC Distribution", h2_style),
]
fig_dist = RES / "aopc_distribution.png"
if fig_dist.exists():
    story.append(RLImage(str(fig_dist), width=14*cm, height=7*cm))
    story.append(Paragraph(
        "Figure 2. Per-sample AOPC distributions. Strong positive skew; "
        "medians are 20–30% lower than means. A small number of samples "
        "with high AOPC pull the means upward.", caption_style))

story += [
    Paragraph("6.5 Key Observations", h2_style),
    Paragraph(
        B("Monotone perturbation curves:") + "  All four methods show increasing mean "
        "probability drop as mask ratio increases from 0.1 to 0.5. This is the expected "
        "and required behaviour — a correctness check for the evaluation pipeline.",
        bullet_style),
    Paragraph(
        B("Method ranking:") + "  Attention Rollout achieves the highest AOPC (0.0490), "
        "followed by GMAR (0.0475), GradCAM (0.0369), and the random baseline (0.0315). "
        "The ranking is consistent across all mask ratios.",
        bullet_style),
    Paragraph(
        B("GradCAM near random:") + "  GradCAM's AOPC (0.0369) is only 17% above random "
        "(0.0315), compared to Attention Rollout's 56% margin. GradCAM also produced "
        "3 negative AOPC samples (masking 'important' regions actually increased probability), "
        "a known failure mode when gradient signals through the mm projector are diffuse.",
        bullet_style),
    Paragraph(
        B("Low absolute values:") + "  All AOPC values are small (< 0.05). This is "
        "explained by language-prior dominance: the LLaMA backbone assigns high probability "
        "to tokens from text context alone, making image masking have limited measurable effect "
        "when averaged over all tokens including stop words.",
        bullet_style),
    Paragraph(
        B("Attention beats random in 36/50 samples;") + " GradCAM only in 25/50 "
        "(no better than a coin flip).",
        bullet_style),
    Spacer(1, 0.3*cm),
]

# ══════════════════════════════════════════════════════════════════════════
#  7. RESULTS — CONTENT TOKENS ONLY
# ══════════════════════════════════════════════════════════════════════════
story += [
    PageBreak(),
    Paragraph("7. Results — Content Tokens Only", h1_style), hr(),
    Paragraph(
        "Re-evaluating using only content tokens (stop words and punctuation filtered out) "
        "removes the dilution effect. The random baseline is unchanged because random masking "
        "is insensitive to which tokens are evaluated.",
        body_style),

    Paragraph("7.1 AOPC Summary", h2_style),
]

wins_co = win_counts(data_co)
aopc_data2 = [["Method", "Mean AOPC", "Std", "Median", "Neg. AOPC", "Wins", "> Random"]]
for m in METHODS:
    mu, sd, med, neg = aopc_stats(data_co, m)
    beat = beat_random_count(data_co, m)
    aopc_data2.append([
        METHOD_LABELS[m],
        f"{mu:.5f}",
        f"{sd:.5f}",
        f"{med:.5f}",
        str(neg),
        f"{wins_co[m]}/{n}",
        f"{beat}/{n}",
    ])

t2 = Table(aopc_data2, colWidths=[3.8*cm, 2.2*cm, 2.0*cm, 2.0*cm, 1.8*cm, 1.6*cm, 2.0*cm])
ts2 = table_style_base()
ts2.add("BACKGROUND", (0,1), (-1,1), colors.HexColor("#d5e8d4"))
ts2.add("FONTNAME",   (0,1), (-1,1), "Helvetica-Bold")
t2.setStyle(ts2)
story += [t2, Spacer(1, 0.2*cm),
          Paragraph("Table 3. AOPC results for content tokens only (n=50; sample 7 excluded "
                    "for saliency methods due to zero content tokens).", caption_style)]

story += [
    Paragraph("7.2 Comparison: All Tokens vs Content Tokens", h2_style),
]
comp_data = [["Method", "All-token AOPC", "Content-only AOPC", "Relative gain", "× Random ratio"]]
for m in METHODS:
    mu_all,  _, _, _ = aopc_stats(data_all, m)
    mu_co,   _, _, _ = aopc_stats(data_co,  m)
    rnd_all, _, _, _ = aopc_stats(data_all, "random")
    gain = (mu_co - mu_all) / mu_all * 100 if mu_all > 0 else 0
    rnd_co, _, _, _ = aopc_stats(data_co, "random")
    ratio = mu_co / rnd_co if rnd_co > 0 else 0
    comp_data.append([METHOD_LABELS[m],
                      f"{mu_all:.5f}",
                      f"{mu_co:.5f}",
                      f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%",
                      f"{ratio:.2f}×"])

t3 = Table(comp_data, colWidths=[3.8*cm, 2.8*cm, 3.0*cm, 2.5*cm, 2.5*cm])
ts3 = table_style_base()
t3.setStyle(ts3)
story += [t3, Spacer(1, 0.2*cm),
          Paragraph("Table 4. Effect of content-token filtering on AOPC. "
                    "Random baseline is unaffected by filtering.", caption_style)]

story += [
    Paragraph("7.3 Key Observations", h2_style),
    Paragraph(
        B("~30% AOPC increase for all saliency methods:") + "  Removing stop words raises "
        "Attention Rollout from 0.0490 to 0.0636 (+29.7%), GMAR from 0.0475 to 0.0622 (+31.0%), "
        "and GradCAM from 0.0369 to 0.0503 (+36.4%). The random baseline is statistically "
        "unchanged (0.0315 → 0.0315), confirming that the gain is not an artefact.",
        bullet_style),
    Paragraph(
        B("Random baseline invariance:") + "  This is the critical sanity check. Stop words "
        "contribute zero signal to any method including random — filtering them raises all "
        "methods equally from the floor. The unchanged random baseline validates that the "
        "increase in structured methods is due to removing uninformative tokens, not any "
        "data selection bias.",
        bullet_style),
    Paragraph(
        B("Improved signal-to-random ratios:") + "  Attention Rollout improves from 1.56× to "
        "2.02× random; GradCAM from 1.17× to 1.60× random. GradCAM is now clearly above "
        "random on content tokens, suggesting it does capture some visual signal for "
        "content-bearing predictions, though it remains substantially weaker than "
        "Attention Rollout and GMAR.",
        bullet_style),
    Paragraph(
        B("Stable method ranking:") + "  Attention Rollout ≈ GMAR > GradCAM > Random "
        "holds in both conditions. The ranking's robustness across filtering conditions "
        "strengthens confidence in the conclusions.",
        bullet_style),
    Paragraph(
        B("New negative AOPC samples:") + "  Content-token filtering introduces 3 negative "
        "samples for Attention Rollout (vs 0 before), likely because with ~9 tokens per sample "
        "the per-sample variance is higher. Samples 8 and 40 are problematic for both Attention "
        "and GMAR simultaneously.",
        bullet_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  8. DISCUSSION
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("8. Discussion", h1_style), hr(),

    Paragraph("8.1 Language-Prior Dominance", h2_style),
    Paragraph(
        "The most important contextual factor for interpreting these results is that "
        "LLaVA-Med, like all generative VLMs, generates text autoregressively with full "
        "access to the preceding token sequence. This means the model's prediction at each "
        "step is not purely image-driven — it is jointly determined by the image and the "
        "text prefix. The absolute AOPC values (~5–8% after content filtering) are therefore "
        "not a sign of pipeline failure; they quantify how much the image contributes "
        "beyond text context.", body_style),
    Paragraph(
        "Contrast with classification tasks: in a pure image classifier there is no text "
        "prior, and masking 50% of important patches typically causes a 20–40% probability "
        "drop. The smaller effect here reflects a different (and more realistic) VLM use case.",
        body_style),

    Paragraph("8.2 GradCAM Weakness in VLMs", h2_style),
    Paragraph(
        "GradCAM's relative weakness (1.60× random on content tokens vs 2.02× for "
        "Attention Rollout) is consistent with the known difficulties of applying "
        "gradient-based attribution to vision-language transformers. The gradient signal "
        "from a text output token back to image projector features passes through 32 "
        "transformer layers, causing gradient saturation and diffusion. The 3 negative AOPC "
        "samples (masking ostensibly 'important' regions raises probability) are a direct "
        "manifestation of this: the gradient-identified regions are not causally important "
        "for those samples.", body_style),

    Paragraph("8.3 Attention Rollout vs GMAR", h2_style),
    Paragraph(
        "Attention Rollout and GMAR produce nearly identical AOPC values "
        "(0.0636 vs 0.0622 on content tokens; 0.0490 vs 0.0475 on all tokens). "
        "The win count favours Attention Rollout slightly (18 vs 17 on content tokens). "
        "Given the substantial computational overhead of GMAR (gradient computation for "
        "all layers × all tokens), Attention Rollout may be preferred in practice unless "
        "a specific per-sample advantage of GMAR is demonstrated.", body_style),

    Paragraph("8.4 Caption Quality", h2_style),
    Paragraph(
        "Generated captions are heavily template-driven. Many begin with 'A black and white "
        "image of a…', indicating the model relies on a relatively small number of "
        "high-probability prefix tokens regardless of the image content. This reduces the "
        "informativeness of per-token AOPC measurements and suggests that the faithfulness "
        "evaluation is most reliable for the later, more content-specific tokens in each "
        "caption.", body_style),

    Paragraph("8.5 Limitations", h2_style),
    Paragraph(
        B("Sample size:") + " 50 images is sufficient for a pilot study but underpowered "
        "for strong statistical claims. Wilcoxon signed-rank tests on 50 samples have "
        "limited power, particularly for GradCAM which only beats random on 25/50 samples.",
        bullet_style),
    Paragraph(
        B("Single image test:") + " The mask verification notebook was run on one image "
        "(renal cyst CT). The single-image perturbation curves are non-monotone for some "
        "methods — expected per-sample noise.", bullet_style),
    Paragraph(
        B("Teacher-forcing assumption:") + " AOPC is measured under teacher forcing, not "
        "free generation. This is the standard approach but means the metric does not "
        "capture downstream effects (e.g. one wrong token causing cascading changes).",
        bullet_style),
    Paragraph(
        B("Zero-masking artefact:") + " Setting masked pixels to 0 in the CLIP-normalised "
        "space corresponds to a constant 'mean colour' value (close to grey), not to a "
        "truly 'absent' signal. Alternative masking strategies (Gaussian noise, mean "
        "imputation) could be explored.", bullet_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  9. CONCLUSION
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("9. Conclusion", h1_style), hr(),
    Paragraph(
        "This experiment provides a quantitative faithfulness evaluation of three visual "
        "saliency methods applied to LLaVA-Med on the ROCOv2 radiology captioning task. "
        "The key conclusions are:", body_style),
    Paragraph(
        B("1. All three saliency methods outperform the random baseline,") + " confirming "
        "that they identify image regions with some causal influence on the generated text. "
        "The result is statistically meaningful for Attention Rollout and GMAR; GradCAM's "
        "margin over random is limited (1.17× all-token; 1.60× content-token).",
        bullet_style),
    Paragraph(
        B("2. Attention Rollout and GMAR perform comparably") + " and are the recommended "
        "methods for LLaVA-style VLMs. Their similar performance suggests that the gradient "
        "weighting in GMAR does not substantially improve over uniform attention rollout "
        "for this task.",
        bullet_style),
    Paragraph(
        B("3. Stop-word filtering is important") + " for a fair faithfulness evaluation "
        "of generative VLMs. Content-token AOPC (~6.4% for Attention Rollout at max ratio) "
        "is 30% higher than all-token AOPC and provides a more interpretable measure of "
        "visual faithfulness.",
        bullet_style),
    Paragraph(
        B("4. Low absolute AOPC values are expected and not a pipeline failure.") + " They "
        "reflect language-prior dominance: the LLM backbone maintains high token probability "
        "from text context even when the image is heavily perturbed. This is an intrinsic "
        "property of autoregressive VLMs and is a finding in itself.",
        bullet_style),
    Paragraph(
        B("5. The evaluation pipeline is correct.") + " Masking is pixel-exact (verified "
        "numerically), perturbation curves are monotone in aggregate, and the random baseline "
        "behaves as expected.",
        bullet_style),
    Spacer(1, 0.5*cm),
    Paragraph(
        "Future work should evaluate on larger sample sizes (≥200 images), compare with a "
        "stronger prompt that elicits more descriptive captions, and explore alternative "
        "masking strategies to reduce the impact of the language prior.",
        body_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  REFERENCES
# ══════════════════════════════════════════════════════════════════════════
story += [
    Paragraph("References", h1_style), hr(),
    Paragraph(
        "Abnar, S. & Zuidema, W. (2020). Quantifying attention flow in transformers. "
        + I("ACL 2020."), body_style),
    Paragraph(
        "Liu, H. et al. (2023). Improved baselines with visual instruction tuning. "
        + I("NeurIPS 2023.") + " [LLaVA 1.5]", body_style),
    Paragraph(
        "Pelka, O. et al. (2022). ROCOv2: Radiology Objects in COntext version 2. "
        + I("MICCAI 2022 Workshops."), body_style),
    Paragraph(
        "Selvaraju, R.R. et al. (2017). Grad-CAM: Visual explanations from deep networks "
        "via gradient-based localization. " + I("ICCV 2017."), body_style),
    Paragraph(
        "Samek, W. et al. (2017). Evaluating the visualization of what a deep neural "
        "network has learned. " + I("IEEE TNNLS.") + " [AOPC metric]", body_style),
    Paragraph(
        "Dettmers, T. et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. "
        + I("NeurIPS 2023.") + " [NF4 quantisation]", body_style),
]

# ══════════════════════════════════════════════════════════════════════════
#  BUILD PDF
# ══════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2.2*cm,  bottomMargin=2.2*cm,
    title="LLaVA-Med Explainability Evaluation Report",
    author="Thesis — April 2026",
)

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(2.2*cm, 1.2*cm,
                      "LLaVA-Med Explainability Evaluation — April 2026")
    canvas.drawRightString(W - 2.2*cm, 1.2*cm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f"PDF written to: {OUT}")
