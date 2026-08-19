# TUTORIAL — Train on a local English corpus (copy/echo demo)

`transformer_lab` is, by default, a sequence-to-sequence **machine translation**
workbench. But you feed it **any** plain-text corpus via an extra `copy` mode: the
encoder and decoder both see the *same* sentence (`src == tgt`), so the model learns
to **reproduce / echo** its input. This lets you demonstrate the full training →
tokenize → decode → evaluate loop on a single file (`sample_corpus.txt`, Grimms'
Fairy Tales) without needing parallel translation data.

## 1. Overview

- Data mode `dataset: "copy"` in `configs/copy_en_en.yaml`.
- `data/corpus.py` reads the text file, drops blank lines / ALL-CAPS headings /
  over-short or over-long lines, and makes `{"src": line, "tgt": line}` records.
- A SentencePiece (BPE, shared vocab ~4k) tokenizer is built from the train split.
- The standard base Transformer (N=6, d_model=512, d_ff=2048, 8 heads) trains with
  the Noam schedule and label smoothing 0.0.
- Device is resolved at runtime (`cuda` → `mps` → `cpu`) with graceful fallback.
  `runtime.device` in the YAML picks the backend; set it to `"cpu"` or `"mps"` as
  needed. AMP is only used on CUDA.

## 2. Setup

```bash
python3 -m pip install -r requirements.txt   # torch, sentencepiece, datasets, sacrebleu, ...
```

(Requires `torch>=2.2`, `datasets`, `sentencepiece`, `sacrebleu`, `loguru`,
`tensorboard`, `tqdm`, `pyyaml`.)

## 3. Train

```bash
python3 train.py --config configs/copy_en_en.yaml
```

What happens:
- `configs/copy_en_en.yaml` selects `dataset: copy` and `corpus_path: sample_corpus.txt`.
- Tokenizer is trained to `work/tokenizer_en_en/spm_4000.model` (cached after the
  first run).
- Batches are built dynamically to ~20k tokens each.
- Checkpoints: `work/checkpoints_en_en/step*.pt` and `work/checkpoints_en_en/best.pt`
  (best validation loss).
- Logs / tensorbboard under `work/logs_en_en` / `work/tb_en_en`.

`max_steps` in the config bounds the run; a few thousand steps on MPS/CPU is enough
to see the copy task converge toward low perplexity.

## 4. Test / echo

```bash
python3 decode.py --config configs/copy_en_en.yaml \
  --ckpt work/checkpoints_en_en/best.pt \
  --src "The king had a beautiful daughter with long golden hair."
```

The output should closely reproduce the input sentence (beam search, size 4,
length penalty 0.6). Minor differences on unseen phrasing are expected early in
training.

## 5. Evaluate (BLEU of reconstruction)

```bash
python3 evaluate.py --config configs/copy_en_en.yaml \
  --ckpt work/checkpoints_en_en/best.pt --split test
```

This runs beam search on the held-out `test` split and reports sacreBLEU of the
echo output vs. the original sentences (a “reconstruction” sanity metric).

## 6. One-command demo

```bash
bash scripts/run_copy_en_en.sh
```

Installs deps (if needed), trains, then decodes three example sentences so you can
see the echo behavior end-to-end.

## 7. Still using the original MT tasks?

Unchanged. `copy` is an **additive** dataset mode; the WMT configs
(`configs/base_en_de.yaml`, `configs/big_en_fr.yaml`, …) still load parallel data
from the Hugging Face hub exactly as before. Only the device handling became
runtime-resolved and therefore safer on Macs.