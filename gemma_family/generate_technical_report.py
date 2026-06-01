"""Generate a 2-page technical report PDF describing the implementation."""
from fpdf import FPDF

class Report(FPDF):
    def header(self):
        if self.page_no() > 0:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Technical Implementation Report -- VLM Saliency & Faithfulness Evaluation", align="C")
            self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(20, 20, 100)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 20, 100)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(60, 60, 60)
        self.cell(0, 5, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4.2, text)
        self.ln(1)

    def mono_text(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(240, 240, 240)
        self.multi_cell(0, 4, text, fill=True)
        self.ln(1)
        self.set_font("Helvetica", "", 9)


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=14)
pdf.set_margins(14, 14, 14)

# ── PAGE 1 ──────────────────────────────────────────────────────────────
pdf.add_page()

pdf.set_font("Helvetica", "B", 14)
pdf.set_text_color(10, 10, 80)
pdf.cell(0, 8, "Technical Implementation: Saliency Extraction & Faithfulness Evaluation", align="C", new_x="LMARGIN", new_y="NEXT")

pdf.set_text_color(80, 80, 80)
pdf.set_font("Helvetica", "I", 9)
pdf.cell(0, 5, "MedGemma 4B (google/medgemma-1.5-4b-it) -- Gemma-3 Vision-Language Architecture", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_text_color(0, 0, 0)

# ── 1. MODEL AND INPUTS ────────────────────────────────────────────────
pdf.section_title("1. Model Interface & Input Construction")

pdf.body_text(
    "The pipeline operates on MedGemma (google/medgemma-1.5-4b-it), a Gemma-3 vision-language model loaded via "
    "HuggingFace Transformers (AutoModelForImageTextToText). It is loaded in NF4 4-bit quantisation "
    "(BitsAndBytesConfig with double quantisation, float16 compute dtype) with attn_implementation=\"eager\" -- "
    "eager attention is mandatory so the model returns per-head attention matrices rather than fused SDPA outputs."
)

pdf.body_text(
    "Input preparation: each image is paired with a clinical prompt and formatted through the processor's chat template. "
    "The processor tokenises the text and encodes the image into pixel_values [1, 3, H, W]. Inside the model, a SigLIP "
    "vision encoder produces 64x64 = 4096 patch embeddings, which are pooled down to 16x16 = 256 image tokens by the "
    "multi_modal_projector. These 256 image tokens are interleaved into the text token sequence at positions identified "
    "via token_type_ids == 1 (or input_ids == 262144)."
)

pdf.body_text(
    "Caption generation uses greedy decoding (do_sample=False) with up to 64 new tokens. The full sequence "
    "[prompt | generated] is preserved as generated_ids. A teacher-forcing input dict is built by setting input_ids = "
    "generated_ids and attention_mask = ones, allowing the model to process the complete sequence in one forward pass "
    "and produce logits at every position simultaneously."
)

# ── 2. ATTENTION SALIENCY ──────────────────────────────────────────────
pdf.section_title("2. Attention Saliency (attention.py)")

pdf.body_text(
    "A single teacher-forcing forward pass is executed with output_attentions=True under torch.no_grad(). This returns "
    "a tuple of attention tensors, one per decoder layer, each of shape [1, H, S, S] where H is the number of attention "
    "heads and S is the total sequence length (prompt + generated tokens)."
)

pdf.body_text(
    "Layer selection: the config specifies global_attn_layers = [5, 11, 17, 23, 29] -- these are the global attention "
    "layers in the Gemma-3 architecture (which alternates local sliding-window and global full-sequence attention). "
    "Only global layers are used because local layers have a limited context window and may not attend to distant image "
    "tokens. The attention_layer_strategy parameter controls selection: \"global\" uses the predefined list, \"all\" uses "
    "every layer, \"lastN\" takes the last N global layers."
)

pdf.body_text(
    "For each generated token at position pos, the method reads attn[0, :, pos, :] -- the attention distribution from "
    "that token across all positions -- then indexes into the image-token positions to get a vector of shape [H, 256]. "
    "This is averaged across heads (axis=0) and then across selected layers to yield a single 256-element vector. "
    "The vector is reshaped to 16x16 and min-max normalised to [0, 1]. Result: one 16x16 saliency map per generated token."
)

# ── 3. GRAD-CAM SALIENCY ───────────────────────────────────────────────
pdf.section_title("3. Grad-CAM Saliency (gradcam.py)")

pdf.body_text(
    "A forward hook (_ProjectorHook) is registered on model.model.multi_modal_projector -- the linear projection layer "
    "that maps SigLIP vision features into the language model's embedding space. The hook intercepts the projector output "
    "tensor of shape [1, 256, D] (where D is the LM hidden dimension, 2560 for this model), detaches it from the "
    "existing computation graph, calls requires_grad_(True) and retain_grad(), then returns it as a replacement. "
    "This makes the projector output a differentiable leaf tensor."
)

pdf.body_text(
    "A teacher-forcing forward pass produces logits [1, S, V]. For each generated token at position pos, the target "
    "token id t is read from generated_ids[0, pos]. The scalar score = logits[0, pos-1, t] (the logit at the preceding "
    "position that predicts token t) is differentiated via torch.autograd.grad w.r.t. the hooked activation. "
    "retain_graph=True is used for all tokens except the last. This yields gradient tensor grad of shape [1, 256, D]."
)

pdf.body_text(
    "Standard Grad-CAM weighting is applied: alpha = grad.mean(dim=1, keepdim=True) -- global average pooling of the "
    "gradient over the spatial (256 image tokens) dimension, yielding shape [1, 1, D]. The class activation map is "
    "cam = ReLU( (alpha * activation).sum(dim=-1) ) of shape [1, 256]. This 256-element vector is reshaped to 16x16 "
    "and min-max normalised. Result: one 16x16 saliency map per generated token, from N backward passes."
)

# ── PAGE 2 ──────────────────────────────────────────────────────────────
pdf.add_page()

# ── 4. GMAR SALIENCY ───────────────────────────────────────────────────
pdf.section_title("4. Gradient-Weighted Model Attention Rollout -- GMAR (gmar.py)")

pdf.subsection_title("Step 1: Gradient-weighted head importance")
pdf.body_text(
    "A teacher-forcing forward pass with output_attentions=True produces both logits and per-layer attention tensors "
    "(gradients enabled). For each generated token at position pos, the target logit score = logits[0, pos-1, t] is "
    "differentiated w.r.t. all attention matrices simultaneously via torch.autograd.grad. For each layer l and each "
    "head h, the per-head importance is computed as the mean absolute gradient: imp(l,h) = mean(|grad[l][0, h, :, :]|). "
    "These importance scores are summed across all generated tokens. Per layer, negative values are clamped to zero and "
    "the vector is L1-normalised so that head weights sum to 1 within each layer."
)

pdf.subsection_title("Step 2: Gradient-weighted attention rollout")
pdf.body_text(
    "Starting from rollout R = I (identity matrix of size S x S), the algorithm iterates over selected layers (same "
    "global-layer selection as the attention method). At each layer l: (a) the attention matrix attn[l][0] of shape "
    "[H, S, S] is combined into a single [S, S] matrix via weighted sum A = sum_h( w_h * attn_h ) using the head "
    "weights from Step 1; (b) a residual connection is applied: A = 0.5*A + 0.5*I, modelling the skip connection in "
    "the transformer; (c) rows are re-normalised to sum to 1; (d) the rollout is updated: R = A @ R. After all layers, "
    "R[i, j] approximates the total effective attention from token i to token j through the entire network."
)

pdf.subsection_title("Step 3: Map extraction")
pdf.body_text(
    "For each generated token at position pos, the row R[pos, :] is indexed at the 256 image-token positions, reshaped "
    "to 16x16, and min-max normalised. This single rollout computation is reused for all generated tokens (no per-token "
    "backward pass for the rollout itself -- the backward passes only serve head weighting)."
)

# ── 5. EVALUATION: PERTURBATION-BASED FAITHFULNESS ─────────────────────
pdf.section_title("5. Perturbation-Based Faithfulness Evaluation & AOPC Metric")

pdf.subsection_title("Baseline token probabilities")
pdf.body_text(
    "Before perturbation, the original token probabilities are obtained: a teacher-forcing forward pass produces logits "
    "[1, S, V]. For each generated token at position pos with ground-truth id t, the probability is "
    "p_orig(pos) = softmax(logits[0, pos-1, :])[t]. These serve as the reference against which probability drops "
    "are measured."
)

pdf.subsection_title("Image masking procedure")
pdf.body_text(
    "Given a 16x16 saliency map and a mask_ratio r in {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}, the top ceil(256*r) grid cells "
    "by saliency value are identified via argsort. A binary mask of shape [1, 1, 16, 16] is created with 1 at those "
    "positions. This is upsampled to pixel resolution [1, 1, H, W] via nearest-neighbour interpolation and applied: "
    "masked_pixels = pixel_values * (1 - mask). The masked image is embedded into the same teacher-forcing input dict "
    "(only pixel_values is replaced), and a new forward pass yields masked probabilities p_masked(pos)."
)

pdf.subsection_title("Evaluation modes")
pdf.body_text(
    "Per-token mode: each generated token's own saliency map is used for masking. This requires one forward pass per "
    "(token, mask_ratio) pair. A content-token filter removes stop words, punctuation, and sub-word fragments (length "
    "<= 1 non-alphabetic), evaluating only semantically meaningful tokens to reduce compute and noise. "
    "Average mode: all per-token saliency maps are stacked and averaged into a single 16x16 map (re-normalised to "
    "[0, 1]), and one forward pass per mask_ratio evaluates all tokens at once."
)

pdf.subsection_title("AOPC computation (Area Over the Perturbation Curve)")
pdf.body_text(
    "For each (token, mask_ratio) pair, the probability drop is: drop = p_orig - p_masked. "
    "For each mask_ratio r, the mean drop across all evaluated tokens is computed: mean_drop(r) = (1/N) * sum(drops). "
    "The AOPC is then the average of these mean drops across all mask ratios:"
)
pdf.mono_text(
    "  AOPC = (1/|R|) * SUM over r in R of [ mean_drop(r) ]\n"
    "       = (1/6) * [ mean_drop(0.1) + mean_drop(0.2) + ... + mean_drop(0.9) ]"
)
pdf.body_text(
    "A higher AOPC indicates that the saliency method correctly identifies regions whose removal causes the largest "
    "probability decrease -- i.e., it is more faithful to the model's actual decision process. A random-baseline "
    "evaluation is also computed using uniformly random saliency maps (seeded for reproducibility); its AOPC serves "
    "as a lower bound."
)

pdf.subsection_title("Per-token output structure")
pdf.body_text(
    "Every evaluation record stores: token position, token id, mask_ratio, original probability, masked probability, "
    "and probability drop. These are aggregated into mean_drops_by_ratio (a dict mapping each ratio to its mean drop) "
    "and the scalar AOPC. Results are saved to all_results.json and summary.csv per run, alongside per-sample "
    "visualisations (saliency grid overlays, method comparison figures, and perturbation curves)."
)


# ── SAVE ────────────────────────────────────────────────────────────────
out_path = "technical_report.pdf"
pdf.output(out_path)
print(f"PDF saved to: {out_path}")
