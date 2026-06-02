# Word-level analysis

Takes the per-token perturbation drops and turns them into the word-level results
from the thesis.

`medgemma.ipynb` and `llava_med.ipynb` do the main experiment: reconstruct words
from sub-word tokens (a word's drop = its first sub-token's), label each word with
`Clinical-AI-Apollo/Medical-NER`, then compute AOPC by method and mask ratio, the
per-word drop distribution, the Friedman test per `(mask ratio, NER class)` with
Benjamini-Hochberg correction, and AOPC by NER class.

`ablation.ipynb` does the 2x2 ablation (Table 2): attention-rollout AOPC for each
medical VLM and its base on ROCOv2 vs MS-COCO. ROCOv2 uses the medical NER (keep
words that hit an entity); COCO drops punctuation and stopwords instead, and uses
the first 100 samples.

Shared code is in `wordlevel.py`. Data is in `data/` (main runs) and
`data/ablation/` (the COCO and base-model conditions).

## Running

```
pip install -r requirements.txt
jupyter notebook
```

Reconstruction needs each model's tokenizer (`google/medgemma-1.5-4b-it`,
`microsoft/llava-med-v1.5-mistral-7b`, `llava-hf/llava-1.5-7b-hf`), so once again you need a hugging face api key.
