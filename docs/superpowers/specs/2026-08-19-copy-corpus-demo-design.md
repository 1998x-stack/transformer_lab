# Design — Train `transformer_lab` on a local monolingual corpus (copy/echo task)

**Date:** 2026-08-19
**Status:** Approved by user (design level); pending spec review.

## Problem

`transformer_lab` is a sequence-to-sequence **machine translation** workbench (modeled on
*Attention Is All You Need*), wired to download parallel corpora from the Hugging Face hub
and hard-coded for `cuda`. The user wants to:

1. Enhance the code and documentation professionally.
2. Train on `sample_corpus.txt` — Grimms' Fairy Tales, a **monolingual English** text.
3. Test with some example sentences.

Because the input corpus is monolingual, the model cannot be trained on a real
translation task with it. The chosen task is a **copy / echo task**: for each corpus
segment, `src == tgt == that segment`, so the encoder–decoder learns to reproduce its
input. Tests then feed the model a sentence and check that it echoes it back.

## Decisions (from brainstorming)

| Question | Decision |
| --- | --- |
| Task on monolingual corpus | **B — copy/echo**: `src = tgt = line` on the existing seq2seq Transformer |
| Training scale | **B — medium**: base config (N=6, d_model=512, d_ff=2048), a few thousand steps, run on **MPS** |
| Code/docs enhancement scope | **C — all three**: (A) local-corpus data path + device-agnostic config, (B) code quality & docs, (C) run script + examples |
| Vocab size | ~4,000 subword tokens (fast on MPS, sufficient for fairy-tale prose) |
| Runtime | Time-bounded few-thousand-step run; stop at a sensible checkpoint rather than looping for hours |

## Architecture & changes

### 1. Data path — local monolingual copy corpus
- New `data/corpus.py`:
  - `parse_copy_corpus(path: str) -> list[dict]`: read lines, strip whitespace, drop
    blank/short/noise lines, and produce `{"src": line, "tgt": line}` entries.
  - Optionally do a deterministic train/validation split (e.g. 95 / 5).
- Extend the data loader in `data/datasets.py` (or add a mode switch):
  - When `DataConfig.dataset == "copy"` and `corpus_path` is set, build the local
    `DatasetDict`; otherwise keep the existing HuggingFace `load_mt_dataset` path.
- **Reuse** `data/tokenization.py` (`SPTokenizer`, `train_or_load_spm`),
  `data/collate.py` (`DynamicBatcher`, `collate`) unchanged — they already operate on
  `{"src", "tgt"}` records.

### 2. Config
- New `configs/copy_en_en.yaml`:
  - `data`: `dataset: "copy"`, `corpus_path: "sample_corpus.txt"`, `lang_pair: "en-en"`,
    `vocab_size: 4000`, small `max_src_len`/`max_tgt_len`, dynamic batching on.
  - `model`: base settings (`N: 6`, `d_model: 512`, `d_ff: 2048`, `num_heads: 8`,
    `label_smoothing: 0.0` — no target-length smoothing for copy, minor override).
  - `optim`: `lr: 5e-4`, `warmup_steps` ~ 400, `max_steps` ~ 5–6k.
  - `runtime`: `device: "mps"`, small `save_every`/`eval_every`/`keep_last`.
  - `decode`: beam 4, length penalty 0.6.

### 3. Device-agnostic trainer
- `train.py` (and `decode.py`/`evaluate.py` where needed): only wrap forward/backward
  in `torch.cuda.amp.autocast` and use `GradScaler` when device is CUDA. On CPU/MPS run
  in fp32. Resolve `cfg.runtime.device` to a real device, handling the "cuda referenced
  but unavailable" case with a clear fallback log.
- This fixes the existing hard `cuda` default so the lab can run on this Mac.

### 4. Code-quality & docs polish (behavior-preserving)
- Type hints + module/class/function docstrings across the touched modules.
- Update `README.md`: add a "Train on your own text (copy demo)" section and a hardware note.
- Add `docs/TUTORIAL.md`: end-to-end walkthrough — install, tokenize, train, decode sample
  sentences, evaluation.

### 5. Run script + examples
- `scripts/run_copy_en_en.sh`: pip install → train → decode a set of example sentences
  (a few held-out validation lines) → print expected vs reproduced text.

### 6. Testing & verification
- `tests/test_smoke.py`: tiny model, few CPU steps, assert loss decreases.
- A copy-decode correctness check (input sentence → model reproduces tokens).
- Then actually run the training on **MPS**, capture checkpoint, and report example
  echo outputs + loss/perplexity (and BLEU of reconstruction as a sanity metric).

## Data flow

```
sample_corpus.txt --parse_copy_corpus--> DatasetDict{train, validation}
    --> train_or_load_spm (shared vocab ~4k) --> DynamicBatcher
    --> collate (src_ids, tgt_in_ids, tgt_out_ids)
    --> Transformer (encode src, decode tgt_in) --> logits
    --> LabelSmoothingLoss -> backward -> Noam scheduler
validation/held-out lines --decode.py beam_search--> echoed text
```

## Error handling & robustness
- Corpus parsing skips malformed/blank/very short lines defensively.
- Device resolution logs and falls back when the configured device is unavailable.
- Checkpoint save uses existing `keep_last` and `best.pt` logic; stop gracefully at
  `max_steps`.

## Non-goals / out of scope
- Real parallel-machine-linguistics training on `sample_corpus.txt` (not applicable).
- GPU/distributed changes beyond device-agnostic fallback.
- Changing MT behavior for WMT configs.