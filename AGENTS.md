# AGENTS.md — Transformer Lab Engineering & Research Contract

This file governs repository-aware coding agents (Codex, OpenCode, and similar tools). Read it **before making changes**. Keep responses aligned with evidence from source code and actual test output.

## Mission and scope

Build a readable, testable, configurable PyTorch Transformer encoder–decoder research workbench based on Vaswani et al., *Attention Is All You Need* (arXiv:1706.03762). Prioritize **correctness, scientific validity, reproducibility and minimal diffs** over feature count.

Three distinct evidence levels must never be conflated:
1. **Paper result** — values or methods explicitly reported by the authors, with citation.
2. **Repository implementation** — functionality confirmed by current code and tests.
3. **New experiment** — a run with actual data, manifest, checkpoint and metrics.

Generating a YAML file or passing a CPU unit test does **not** reproduce WMT14 BLEU.

## Read-first map

| Area | Files | Why |
| --- | --- | --- |
| Documentation | `README.md`, `docs/README.md`, `docs/ABLATION_RESEARCH.md`, `docs/FACTORY_MODE.md` | Scope, paper details, expected contracts |
| Configuration | `utils/config.py`, `configs/base_en_de.yaml`, `configs/copy_en_en.yaml` | Dataclass schema, defaults and runnable presets |
| Core model | `models/transformer.py`, `models/factory.py` | Encoder/decoder, attention, variant construction |
| Objective & schedule | `models/label_smoothing.py`, `optim/scheduler.py` | Smoothed objective vs hard-target NLL, Noam steps |
| Data | `data/datasets.py`, `data/tokenization.py`, `data/collate.py` | Splits, tokenization, masks, dynamic batches |
| Entrypoints | `train.py`, `evaluate.py`, `decode.py` | Real CLI behavior; model construction goes through factories |
| Research tools | `research/ablations.py`, `research/tracking.py`, `research/visualize.py` | Matrix, provenance, JSONL and plots |
| Tests | `tests/test_factory_ablations.py`, `tests/unit/`, `tests/integration/` | Regression contracts and broader validation |

Historical material under `docs/archive/` and `docs/superpowers/` is reference material, **not** an authority over current code.

## Non-negotiable architecture rules

- **Factory Mode**: instantiate training/evaluation/decoding components via `models.factory`. Extend registries explicitly; do not create scattered `Transformer(...)` branches, use dynamic `eval`, or silently fallback for unknown components.
- Keep **one semantic source of truth** for a hyperparameter in the typed config; configurations must be serializable and reproducible.
- Preserve public `train.py --config`, `evaluate.py --config --ckpt`, `decode.py --config --ckpt --src` interfaces unless a breaking change is explicit and documented.
- Decoder self-attention is **causal**. Boolean masks use `True` = allowed/visible. Check all-masked rows for finite values and test prefix invariance.
- Multi-head attention independently supports `d_k` and `d_v`; key/query projection widths are `num_heads * d_k`, value projection width is `num_heads * d_v`. Do not mislabel a `d_k`-only run as a matched-compute head-count ablation.
- Distinguish historical architecture compatibility (extra final LayerNorm and FFN inner dropout) from paper-faithful settings explicitly enabled in the `paper` experiment matrix.
- Do not hide incompatible checkpoint shapes with `strict=False`.

## Scientific experiment protocol

Before implementing or claiming an ablation:
1. Identify its **paper section/table/row** (if applicable), hypothesis, controlled variables, and deviations from the original method.
2. Generate full, immutable **per-run YAML** rather than relying on partial override examples from `configs/ablations/`.
3. Record seed, code commit, config fingerprint, dataset revision/splits, tokenizer fingerprint, device, training budget and metric protocol. If current instrumentation cannot record one of these, **mark it missing**; do not invent it.
4. Keep paper-matching variants separate from `mechanism` controls (e.g., Pre-LN, no scaling, GELU). Avoid confounding changed model capacity or token budgets.
5. Evaluate with the same splits, tokenization, decoding and metric signatures. Prefer **hard-target NLL for PPL**, not the smoothed training loss; count valid non-pad target tokens.
6. Test determinism, seed separation and no train/validation/test leakage. Reserve held-out test data for final comparisons.
7. For comparative claims, report actual run count, seeds, mean/variance, parameter count and environment, along with failures.

`copy` is an engineering sanity test, not an En↔De / En↔Fr WMT14 reproduction. Exact comparison with the paper requires original data processing, training schedule, checkpoint and decoding methodology.

## Change workflow

1. **Inspect before editing:** read existing files/tests, find consumers, and choose the smallest coherent change. Avoid broad renames or deletions without tracing references.
2. **Plan contracts:** describe changed module interfaces, experiment factors, target metrics and required regressions.
3. **Implement:** Python 3.10+ with types where practical; no needless new dependencies; log with the project's Loguru/JSONL/TensorBoard conventions.
4. **Verify:** run the narrowest relevant test first, then broader checks as allowed by compute/data availability. Do not silently execute huge WMT training or download giant datasets.
5. **Document:** keep README commands, config keys, `docs/ABLATION_RESEARCH.md`, and `docs/FACTORY_MODE.md` synchronized. Put historical discussions in `docs/archive/`, not repository root.
6. **Deliver:** report exactly what changed, where, which commands were actually run, pass/fail/skip results, any scientific deviations, and remaining blockers.

## Local command contracts

From the **repository root**:

```bash
python -m pip install -r requirements.txt
python -m compileall -q models optim research utils train.py evaluate.py decode.py
python -m pytest -q -o addopts='' tests/test_factory_ablations.py
python -m pytest -q                     # broader suite; may need optional data/deps

python -m research.ablations \
  --base configs/base_en_de.yaml \
  --out work/ablations \
  --suite paper --seeds 42 43 44
python train.py --config work/ablations/configs/B_dk_16-s42.yaml

# Only after training records exist
python -m research.visualize work/ablations/logs/*/* --output work/ablations/plots
```

The repository uses root-level scripts (`train.py`, `evaluate.py`, `decode.py`); do **not** assume that `python -m transformer_lab.train` resolves from this layout.

## Test and review checklist

- [ ] Model tensors, masking, forward/backward, and finite outputs covered; causal prefix unchanged by future target tokens.
- [ ] `d_k`/`d_v`, attention head counts, optional normalization/position modes and tied weights covered.
- [ ] Learning-rate schedule, gradient accumulation and PAD-ignore behavior verified.
- [ ] No paper result is presented as locally measured; new plots consume real `metrics.jsonl` only.
- [ ] No large datasets, tokenizers, model weights, private corpora, logs, `.env`, or credentials committed.
- [ ] README, AGENTS, docs, configs and actual CLI continue to agree.
- [ ] Validation scope stated precisely (unit, integration, smoke, BLEU, reproducibility are different).

## Repository hygiene

- Runtime artifacts belong under `work/` (ignored), not tracked in git. `data/` contains **source code**; put local datasets in ignored `data/local/` or another ignored runtime directory.
- Keep experimental plans under `docs/superpowers/`, reference-only notes under `docs/archive/`, runnable tutorials under `docs/`.
- Do not delete historical records, rewrite previous experiments, silently change numerical defaults, or turn placeholder scores into measurements.
- If asked to 'tidy files', prefer links, archives and consistent names over unreviewed wholesale restructuring.

## Definition of done

A change is complete only when its affected code paths are validated, intended reproducibility artifacts are inspectable, documentation is consistent, and open limitations are explicitly listed. When verification is impossible in the current environment, say **not run**, not **passed**.
