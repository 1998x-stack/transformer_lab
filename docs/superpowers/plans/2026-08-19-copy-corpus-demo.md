# Copy-Corpus Demo for `transformer_lab` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `transformer_lab` train on a local monolingual text corpus via a copy/echo task (src == tgt == line), running device-agnostically (CPU/MPS) with professional code/docs, a run script, and example-echo verification.

**Architecture:** Add a `copy` dataset mode that reads `sample_corpus.txt` (Grimms' Fairy Tales) into `src == tgt` sentence records, reusing the existing SPM tokenizer, dynamic batcher, collate, seq2seq Transformer, Noam scheduler, and beam search. Make the trainer/decoder/evaluator resolve the device at runtime (CUDA/MPS/CPU) and only use AMP on CUDA. Ship a small MPS config, a run script, a smoke test, updated README, and a tutorial.

**Tech Stack:** PyTorch 2.8, sentencepiece, datasets 4.5, sacrebleu, loguru, tensorboard, pytest 8.

## Global Constraints

- Package root = repo root; imports are absolute (`from data.corpus import ...`); run from repo root.
- This machine is **CPU + MPS, no CUDA**. `runtime.device` may be `mps`; must fall back to CPU if unavailable.
- Do not edit `sample_corpus.txt` (repo root, Grimms' Fairy Tales).
- Copy demo: **vocab_size 4000**, **base model** N=6 / d_model=512 / d_ff=2048 / heads=8, `label_smoothing: 0.0`.
- The existing WMT/HF-hub MT path must keep working; `copy` is additive.
- Tests run on **CPU** via `pytest tests/ -v`.
- A Python 3.9 environment: nothing needs new language features beyond those already used.
- Every task commits.

---

### Task 0: Install test/build dependencies

**Files:** none.

- [ ] **Step 1: Install missing deps**

Run:

```bash
python -m pip install sentencepiece sacrebleu pytest
```

(Verify: `python -c "import sentencepiece, sacrebleu, pytest; print('ok')"` prints `ok`.)
Expected: `ok`.

- [ ] **Step 2: Commit (if any lock/requirements changed)**

No commit needed unless `requirements.txt` was edited (it already lists all deps). Skip commit.

---

### Task 1: Config + device surface

**Files:**
- Modify: `utils/config.py`
- Modify: `utils/torch_utils.py`
- Create: `tests/test_device.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `DataConfig.corpus_path: Optional[str]`; `DataConfig.dataset` accepts `"copy"`.
- Produces: `resolve_device(name: str | None) -> torch.device`.
- Consumes: nothing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_device.py`:

```python
import torch


def test_resolve_cpu():
    from utils.torch_utils import resolve_device

    assert resolve_device("cpu").type == "cpu"


def test_resolve_fallback_on_unavailable_backend():
    from utils.torch_utils import resolve_device

    dev = resolve_device("cuda") if not torch.cuda.is_available() else resolve_device("cpu")
    # On this machine CUDA is unavailable, so resolve_device must fall back, never crash.
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_resolve_none_returns_something():
    from utils.torch_utils import resolve_device

    assert resolve_device(None).type in ("cuda", "mps", "cpu")
```

Create `tests/test_config.py`:

```python
def test_dataconfig_accepts_copy_and_corpus_path():
    from utils.config import DataConfig

    cfg = DataConfig(dataset="copy", corpus_path="sample_corpus.txt")
    assert cfg.dataset == "copy"
    assert cfg.corpus_path == "sample_corpus.txt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_device.py tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests'` NOT acceptable; instead FAIL with `AttributeError`/`TypeError` because `resolve_device` and `corpus_path` don't exist yet. (Note: run `python -m pytest` from repo root so package imports resolve.)

- [ ] **Step 3: Implement**

In `utils/config.py`, change the `dataset` field and add `corpus_path`:

```python
    dataset: Literal["wmt14", "wmt16", "opus100", "copy"] = "wmt14"
```

and append `corpus_path` to `DataConfig` (after `cache_dir`):

```python
    cache_dir: Optional[str] = None
    corpus_path: Optional[str] = None  # used when dataset == "copy"
```

In `utils/torch_utils.py`, append:

```python
def resolve_device(name: str | None) -> torch.device:
    """Resolve a config device name to a real device with backend fallback.

    Falls CUDA -> MPS -> CPU and MPS -> CPU when the requested backend is
    unavailable; auto-picks a backend when ``name`` is ``None``.
    """
    if name is not None:
        if name == "cuda" and not torch.cuda.is_available():
            name = "mps" if torch.backends.mps.is_available() else "cpu"
        elif name == "mps" and not torch.backends.mps.is_available():
            name = "cpu"
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_device.py tests/test_config.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add utils/config.py utils/torch_utils.py tests/test_device.py tests/test_config.py
git commit -m "feat: add copy dataset mode + device resolution with fallback"
```

---

### Task 2: Copy-corpus data path

**Files:**
- Create: `data/corpus.py`
- Modify: `data/datasets.py` (add `load_copy_corpus`)
- Create: `tests/test_corpus.py`

**Interfaces:**
- Produces: `parse_copy_corpus(path: str, min_len: int, max_len: int) -> list[dict]` — records `{"src": text, "tgt": text}`.
- Produces: `load_copy_corpus(corpus_path: str, min_len: int, max_len: int, seed: int = 42, val_ratio: float = 0.05, test_ratio: float = 0.05) -> DatasetDict` — keys `train`/`validation`/`test`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus.py`:

```python
def test_parse_copy_corpus_src_equals_tgt(tmp_path):
    from data.corpus import parse_copy_corpus

    f = tmp_path / "c.txt"
    f.write_text("The quick brown fox jumps over the lazy dog.\n\nFOO BAR\n\nOnce upon a time there was a king.\n")
    recs = parse_copy_corpus(str(f), min_len=4, max_len=100)
    assert len(recs) == 2  # two real sentences; the ALL-CAPS 'FOO BAR' is dropped
    for r in recs:
        assert r["src"] == r["tgt"]
        assert r["src"].strip()


def test_load_copy_corpus_splits(tmp_path):
    from data.datasets import load_copy_corpus

    f = tmp_path / "c.txt"
    lines = [f"line {i} eleven words here to make length" for i in range(100)]
    f.write_text("\n".join(lines) + "\n")
    ddict = load_copy_corpus(str(f), min_len=4, max_len=100, seed=0)
    assert set(ddict.keys()) == {"train", "validation", "test"}
    assert len(ddict["train"]) > 0
    assert sum(len(ddict[k]) for k in ("train", "validation", "test")) == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data.corpus'`.

- [ ] **Step 3: Implement**

Create `data/corpus.py`:

```python
# transformer_lab/data/corpus.py
from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict


def parse_copy_corpus(path: str, min_len: int, max_len: int) -> List[Dict[str, str]]:
    """Read a plain-text file into ``src == tgt`` copy records.

    Skips blank lines, ALL-CAPS headings/dividers, and lines outside
    ``[min_len, max_len]`` words so the demo corpus is clean sentence-like prose.
    """
    records: List[Dict[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = " ".join(raw.split())
        n_words = len(text.split())
        if not text or n_words < min_len or n_words > max_len:
            continue
        if all(c.isupper() or c in " .,;:!?“”‘’'-" for c in text):
            continue
        records.append({"src": text, "tgt": text})
    return records


def load_copy_corpus(
    corpus_path: str,
    min_len: int,
    max_len: int,
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> DatasetDict:
    """Build a deterministic train/validation/test split of copy records."""
    records = parse_copy_corpus(corpus_path, min_len, max_len)
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = records[:n_test]
    val = records[n_test : n_test + n_val]
    train = records[n_test + n_val :]
    return DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add data/corpus.py data/datasets.py tests/test_corpus.py
git commit -m "feat: add copy-corpus parsing and train/val/test split"
```

---

### Task 3: Device-agnostic trainer with copy mode

**Files:**
- Modify: `train.py`

**Interfaces:**
- Consumes: `resolve_device` (Task 1), `load_copy_corpus` (Task 2).
- Produces: `train.py` runs on CPU/MPS and honors `DataConfig.dataset == "copy"`.

- [ ] **Step 1: Verify the file compiles after edit**

Run: `python -c "import ast; ast.parse(open('train.py').read()); print('parse ok')"`
Expected: `parse ok`.

- [ ] **Step 2: Implement — edit the data-loading and device sections**

Make these targeted edits to the existing `train.py`:

**Edit A — imports.** Replace the import block's data/device lines:

```python
from utils.distributed import set_seed, is_main_process
from utils.torch_utils import create_padding_mask, count_parameters
```

becomes:

```python
from utils.distributed import set_seed, is_main_process
from utils.torch_utils import create_padding_mask, count_parameters, resolve_device
```

and add `load_copy_corpus` to the data import:

```python
from data.datasets import load_mt_dataset, filter_and_rename
```

becomes:

```python
from data.datasets import load_mt_dataset, filter_and_rename, load_copy_corpus
```

**Edit B — data loading.** Replace the language-pair parse + dataset-load block:

```python
    # 语言对解析
    src_lang, tgt_lang = cfg.data.lang_pair.split("-")
    # 加载数据
    ddict: DatasetDict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
    ddict = filter_and_rename(ddict, src_lang, tgt_lang, cfg.data.min_len, cfg.data.max_src_len, cfg.data.max_tgt_len)
```

with:

```python
    # 数据：支持本地 copy 语料 (src==tgt) 与 HF 平行语料两种模式
    if cfg.data.dataset == "copy":
        ddict: DatasetDict = load_copy_corpus(cfg.data.corpus_path, cfg.data.min_len, cfg.data.max_src_len, seed=cfg.runtime.seed)
    else:
        src_lang, tgt_lang = cfg.data.lang_pair.split("-")
        ddict: DatasetDict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
        ddict = filter_and_rename(ddict, src_lang, tgt_lang, cfg.data.min_len, cfg.data.max_src_len, cfg.data.max_tgt_len)
```

**Edit C — device + AMP.** Immediately after the data loading, replace:

```python
    # 模型
    model = Transformer(
```

so that before the model, insert:

```python
    device = resolve_device(cfg.runtime.device)
    use_amp = cfg.runtime.amp and device.type == "cuda"
    logger.info(f"Training on device={device} amp={use_amp}")

    # 模型
```

Then replace the model `.to(cfg.runtime.device)` at the end of the constructor call:

```python
    ).to(cfg.runtime.device)
```

becomes:

```python
    ).to(device)
```

**Edit D — optimizer/scaler.** Replace:

```python
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.runtime.amp)
```

with:

```python
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
```

**Edit E — autocast.** Replace:

```python
            with torch.cuda.amp.autocast(enabled=cfg.runtime.amp):
```

with:

```python
            with torch.cuda.amp.autocast(enabled=use_amp):
```

**Edit F — all remaining `.to(cfg.runtime.device)`.** Replace each `...to(cfg.runtime.device)` occurrence in the training and eval inner loops with `.to(device)` and every `path = torch.load(...map_location=cfg.runtime...)` style check. In `train.py` the four `.to(cfg.runtime.device)` calls inside the loops become `.to(device)`.

- [ ] **Step 3: Verify compile**

Run: `python -c "import ast; ast.parse(open('train.py').read()); print('parse ok')"` and `grep -n "runtime.device" train.py` → should now print **no lines** (all replaced).

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat: device-agnostic trainer with copy-corpus mode"
```

---

### Task 4: Device-agnostic decoder & evaluator

**Files:**
- Modify: `decode.py`

**Interfaces:**
- Consumes: `resolve_device` (Task 1), `load_copy_corpus` (Task 2).
- Produces: `decode.py` runs on CPU/MPS and supports the `copy` dataset mode.

- [ ] **Step 1: Edit `decode.py` imports**

Add `resolve_device` and `load_copy_corpus`:

```python
from utils.config import load_config
from data.tokenization import train_or_load_spm
from data.datasets import load_mt_dataset, filter_and_rename
```

becomes:

```python
from utils.config import load_config
from utils.torch_utils import resolve_device
from data.tokenization import train_or_load_spm
from data.datasets import load_mt_dataset, filter_and_rename, load_copy_corpus
```

- [ ] **Step 2: Edit device + data loading in `decode.py`**

Replace:

```python
    cfg = load_config(args.config)
    src_lang, tgt_lang = cfg.data.lang_pair.split("-")
    ddict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
    tok = train_or_load_spm(ddict["train"], cfg.data.tokenizer_dir, cfg.data.vocab_size)
```

with:

```python
    cfg = load_config(args.config)
    device = resolve_device(cfg.runtime.device)
    ddict = (load_copy_corpus(cfg.data.corpus_path, cfg.data.min_len, cfg.data.max_src_len, seed=cfg.runtime.seed)
             if cfg.data.dataset == "copy"
             else load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir))
    tok = train_or_load_spm(ddict["train"], cfg.data.tokenizer_dir, cfg.data.vocab_size)
```

Then replace the two device usages:

```python
    ).to(cfg.runtime.device)
    state = torch.load(args.ckpt, map_location=cfg.runtime.device)
```

with:

```python
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
```

and:

```python
    src_ids = torch.tensor([tok.encode(args.src)], device=cfg.runtime.device)
```

with:

```python
    src_ids = torch.tensor([tok.encode(args.src)], device=device)
```

- [ ] **Step 3: Edit `evaluate.py` equivalently**

Add `resolve_device` and `load_copy_corpus` to imports, then in `main()`:

```python
    cfg = load_config(args.config)
    tb = setup_logging(cfg.runtime.tb_dir + "_eval")
    set_seed(cfg.runtime.seed)

    src_lang, tgt_lang = cfg.data.lang_pair.split("-")
    ddict: DatasetDict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
    ddict = filter_and_rename(ddict, src_lang, tgt_lang, cfg.data.min_len, cfg.data.max_src_len, cfg.data.max_tgt_len)
```

becomes:

```python
    cfg = load_config(args.config)
    tb = setup_logging(cfg.runtime.tb_dir + "_eval")
    set_seed(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)

    if cfg.data.dataset == "copy":
        ddict: DatasetDict = load_copy_corpus(cfg.data.corpus_path, cfg.data.min_len, cfg.data.max_src_len, seed=cfg.runtime.seed)
    else:
        src_lang, tgt_lang = cfg.data.lang_pair.split("-")
        ddict: DatasetDict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
        ddict = filter_and_rename(ddict, src_lang, tgt_lang, cfg.data.min_len, cfg.data.max_src_len, cfg.data.max_tgt_len)
```

Then replace `).to(cfg.runtime.device)` → `).to(device)`, `state = torch.load(args.ckpt, map_location=cfg.runtime.device)` → `map_location=device`, and `src_ids = torch.tensor([tok.encode(src)], device=cfg.runtime.device)` → `device=device`.

- [ ] **Step 4: Verify compile**

Run: `python -c "import ast; [ast.parse(open(f).read()) for f in ('decode.py','evaluate.py')]; print('parse ok')"`
Expected: `parse ok`.

- [ ] **Step 5: Commit**

```bash
git add decode.py evaluate.py
git commit -m "refactor: device-agnostic decode/evaluate with copy mode"
```

---

### Task 5: Copy demo config + run script

**Files:**
- Create: `configs/copy_en_en.yaml`
- Create: `scripts/run_copy_en_en.sh`
- Modify: `tests/test_config.py` (extend)

**Interfaces:**
- Consumes: `DataConfig` fields (Task 1).
- Produces: a runnable copy-demo config and `bash scripts/run_copy_en_en.sh`.

- [ ] **Step 1: Write the config**

Create `configs/copy_en_en.yaml`:

```yaml
model:
  N: 6
  d_model: 512
  d_ff: 2048
  num_heads: 8
  dropout: 0.1
  attn_dropout: 0.0
  activation: "relu"
  share_embeddings: true
  tie_softmax_weight: true
  pos_encoding: "sinusoidal"
  label_smoothing: 0.0

data:
  dataset: "copy"
  lang_pair: "en-en"
  corpus_path: "sample_corpus.txt"
  max_src_len: 128
  max_tgt_len: 128
  min_len: 4
  tokenizer_dir: "work/tokenizer_en_en"
  vocab_size: 4000
  use_shared_vocab: true
  max_tokens_per_batch: 20000
  num_buckets: 8
  cache_dir: null

optim:
  lr: 5.0e-4
  betas: [0.9, 0.98]
  eps: 1.0e-9
  weight_decay: 0.0
  warmup_steps: 400
  max_steps: 6000
  grad_clip: 1.0

runtime:
  seed: 42
  device: "mps"
  num_workers: 1
  log_dir: "work/logs_en_en"
  ckpt_dir: "work/checkpoints_en_en"
  tb_dir: "work/tb_en_en"
  save_every: 1000
  eval_every: 1000
  keep_last: 5
  amp: false
  accumulate_steps: 1

decode:
  beam_size: 4
  length_penalty: 0.6
  max_len_offset: 30
  max_len_ratio: 1.1
```

Note: `max_len_ratio` and `max_len_offset` are passed positionally/otherwise in decode; verify they match `DecodeConfig` field names (max_len_offset, max_len_ratio — yes).

- [ ] **Step 2: Extend `tests/test_config.py`**

Append:

```python
def test_load_copy_config(tmp_path):
    from utils.config import load_config

    cfg = load_config("configs/copy_en_en.yaml")
    assert cfg.data.dataset == "copy"
    assert cfg.data.corpus_path == "sample_corpus.txt"
    assert cfg.model.N == 6
    assert cfg.optim.max_steps == 6000
```

- [ ] **Step 3: Run config test**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (4 passed).

- [ ] **Step 4: Write the run script**

Create `scripts/run_copy_en_en.sh`:

```bash
#!/usr/bin/env bash
# Train the base Transformer on Grimms' Fairy Tales (copy/echo task) and verify it echoes.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> Installing dependencies (if needed)"
python -m pip install -q -r requirements.txt

echo ">> Training copy/echo model on sample_corpus.txt..."
python train.py --config configs/copy_en_en.yaml

echo ">> Echo demo (expected: model reproduces the input sentence)"
examples=(
  "There was once a poor man who lived in the forest."
  "The king had a beautiful daughter with long golden hair."
  "In the morning the little bird began to sing very sweetly."
)
for s in "${examples[@]}"; do
  echo "in : $s"
  out=$(python decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "$s")
  echo "out: $out"
  echo
done
```

Then: `chmod +x scripts/run_copy_en_en.sh`.

- [ ] **Step 5: Commit**

```bash
git add configs/copy_en_en.yaml scripts/run_copy_en_en.sh tests/test_config.py
git commit -m "feat: add copy-corpus demo config and run script"
```

(If `chmod` applied, use `git update-index --chmod=+x scripts/run_copy_en_en.sh` or re-add after chmod.)

---

### Task 6: Smoke test (tiny train loop)

**Files:**
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `parse_copy_corpus` (Task 2), `SPTokenizer`, `LabelSmoothingLoss`, `Transformer`, `collate` (existing).

- [ ] **Step 1: Write the test**

Create `tests/test_smoke.py`:

```python
import torch

from data.collate import Sample, collate
from data.tokenization import SPTokenizer
from models.label_smoothing import LabelSmoothingLoss
from models.transformer import Transformer


def _tiny_tokenizer(tmp_path):
    import sentencepiece as spm

    lines = [f"This is sentence number {i} in the fairy corpus." for i in range(60)]
    f = tmp_path / "c.txt"
    f.write_text("\n".join(lines), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(f), model_prefix=str(tmp_path / "spm"), vocab_size=300,
        character_coverage=1.0, model_type="bpe", bos_id=1, eos_id=2, pad_id=0, unk_id=3,
    )
    return SPTokenizer(str(tmp_path / "spm.model"))


def test_smoke_training_decreases_loss(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    torch.manual_seed(0)
    model = Transformer(
        vocab_size=tok.vocab_size, N=1, d_model=64, d_ff=128, num_heads=4, dropout=0.0,
        attn_dropout=0.0, activation="relu", share_embeddings=True, tie_softmax_weight=True,
        pos_encoding="sinusoidal",
    )
    crit = LabelSmoothingLoss(classes=tok.vocab_size, smoothing=0.0, ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)

    phrases = [
        "this is the first sample sentence",
        "the second line carries more meaning",
        "a third fairy tale begins here",
        "and finally the fourth one ends",
    ]
    samples = [Sample(tok.encode(p), tok.encode(p, add_bos=True, add_eos=True)) for p in phrases]
    batch = collate(samples, pad_id=tok.pad_id)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        pad = tok.pad_id
        src = batch["src_ids"]
        tgt_in = batch["tgt_in_ids"]
        tgt_out = batch["tgt_out_ids"]
        # keep pad-position targets as ignore_index for the loss
        target = tgt_out.masked_fill(tgt_out == pad, -100)
        logits = model(src, tgt_in, src != pad, tgt_in != pad)
        loss = crit(logits, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss must decrease, got {losses[0]} -> {losses[-1]}"
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: PASS (1 test passes; loss monotonically/near-monotonically decreases over 20 updates).

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add smoke test that tiny transformer trains on CPU"
```

---

### Task 7: Docs polish

**Files:**
- Modify: `README.md`
- Create: `docs/TUTORIAL.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Update `README.md`**

Add, after the existing "快速开始" section, a new section "在本地语料上训练（copy/echo 演示）" (or English "Train on your own corpus — copy/echo demo") that:

- Explains that `sample_corpus.txt` is monolingual English, so the lab runs as a copy/echo task (src == tgt).
- Gives commands:

```bash
python -m pip install -r requirements.txt
bash scripts/run_copy_en_en.sh
```

- Explains device auto-fallback (CUDA/MPS/CPU) and how to override via the `runtime.device` key in a YAML config.
- Points to `docs/TUTORIAL.md` for a full walkthrough.

- [ ] **Step 2: Create `docs/TUTORIAL.md`**

Write a concise, complete tutorial with the following sections (each 2–5 paragraphs + code blocks):

1. **Overview** — what the copy demo does and why (seq2seq reconstructs its input).
2. **Setup** — `pip install -r requirements.txt` (includes torch, sentencepiece, datasets, sacrebleu, loguru, tensorboard, tqdm, pyyaml).
3. **Train** — `python train.py --config configs/copy_en_en.yaml`, explain tokenizer at `work/tokenizer_en_en`, batching, checkpoints at `work/checkpoints_en_en/step*.pt` and `best.pt`.
4. **Test / echo** — `python decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "the king had a beautiful daughter"`; explain beam search, that output should match input.
5. **Evaluate** — `python evaluate.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --split test` reports sacreBLEU of reconstruction.
6. **Running on the original MT tasks** — note this is unchanged; the `copy` mode is additive; WMT configs still call the HF hub.

Do not invent numbers — describe the mechanics, not "training achieves X%".

- [ ] **Step 3: Commit**

```bash
git add README.md docs/TUTORIAL.md
git commit -m "docs: add copy-corpus demo docs and tutorial"
```

---

### Task 8: Full run + example verification (manual, not TDD)

**Files:** none (output under `work/` is git-ignored as appropriate).

**Interfaces:** consumes all prior tasks.

- [ ] **Step 1: Run the smoke test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS.

- [ ] **Step 2: Run the copy demo on MPS**

Run: `python train.py --config configs/copy_en_en.yaml`
Expected: training logs; `best.pt` saved in `work/checkpoints_en_en/`; the `copy` corpus loads from `sample_corpus.txt`; loss/perplexity trends downward.

(If MPS is slow, this plan's `maxSteps: 6000` is a guide; a few thousand steps is sufficient to produce a checkpoint that echoes common phrases. Do not let it loop for hours — stop at a sensible checkpoint and record examples.)

- [ ] **Step 3: Echo examples**

Run:

```bash
python decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "There was once a poor man who lived in the forest."
python decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "The king had a beautiful daughter with long golden hair."
python decode.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --src "In the morning the little bird began to sing very sweetly."
```

Expected: each output closely reproduces its input (slight variance on unseen phrasing is acceptable).

- [ ] **Step 4: Optionally evaluate**

Run: `python evaluate.py --config configs/copy_en_en.yaml --ckpt work/checkpoints_en_en/best.pt --split test`
Expected: prints a `sacreBLEU` score for reconstruction.

- [ ] **Step 5: Commit final docs/report**

If useful, add a short note to `docs/TUTORIAL.md`'s run section, else no commit required. Verify the repo is clean otherwise.

---

## Self-Review

**Spec coverage:**
- Copy data path (Task 2) ✔
- `copy_en_en.yaml` config (Task 5) ✔
- Device-agnostic trainer/decode/eval incl. AMP-on-CUDA guard (Tasks 3–4) ✔
- Run script + examples (Tasks 5, 8) ✔
- Smoke test / correctness (Task 6) ✔
- Docs polish — README + tutorial (Task 7) ✔
- Actual MPS run + reported examples (Task 8) ✔

**Type consistency:** `load_copy_corpus(corpus_path, min_len, max_len, seed)` used identically in Tasks 2, 3, 4. `resolve_device(name) -> torch.device` used in Tasks 3–4. `DataConfig.dataset` accepts `"copy"` (Task 1) and is checked in all three entry points. `DecodeConfig` uses `max_len_offset`/`max_len_ratio` consistently.

**Ambiguity resolved:** copy-mode length cap uses `max_src_len` as the single generic cap everywhere (`parse_copy_corpus` max, `load_copy_corpus`, decode/eval branches) to avoid a new `max_len` field.

**Placeholder scan:** no TBD/TODO; every code step has exact code and expected output. Task 8 is explicitly an execution/verification step, not a placeholder.

**Scope:** focused on the single copy-demo feature; MT path untouched except for additive device handling.