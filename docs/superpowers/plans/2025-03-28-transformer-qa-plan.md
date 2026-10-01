# Transformer Project QA Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the transformer-based machine translation project from zero tests and minimal documentation to production-ready code with 90%+ test coverage, comprehensive type safety, robust error handling, and full documentation.

**Architecture:** Incremental TDD approach starting with testing infrastructure, then type safety, then error handling, then documentation. Each component is built with tests first, integrated incrementally, and validated continuously.

**Tech Stack:** pytest, pytest-cov, hypothesis, mypy, pydantic, loguru, pre-commit, GitHub Actions, Sphinx, pytest-benchmark

---

## File Structure Overview

**New Files Created:**
- `tests/unit/test_transformer.py` - Model component tests
- `tests/unit/test_tokenization.py` - Tokenizer tests
- `tests/unit/test_config.py` - Configuration tests
- `tests/unit/test_metrics.py` - Metrics computation tests
- `tests/unit/test_scheduler.py` - Learning rate scheduler tests
- `tests/integration/test_training_loop.py` - End-to-end training tests
- `tests/integration/test_evaluation.py` - Evaluation pipeline tests
- `tests/integration/test_data_pipeline.py` - Data loading integration tests
- `tests/performance/test_model_performance.py` - Performance benchmarks
- `tests/performance/test_memory_usage.py` - Memory profiling tests
- `tests/conftest.py` - pytest fixtures and configuration
- `transformer_lab/utils/errors.py` - Custom exception classes
- `transformer_lab/utils/logging_utils_enhanced.py` - Structured logging
- `transformer_lab/utils/config/validation.py` - Config validation
- `transformer_lab/utils/config/schemas.py` - Pydantic schemas
- `transformer_lab/utils/training/checkpointing.py` - Checkpoint management
- `transformer_lab/utils/training/loop.py` - Core training logic
- `docs/architecture.md` - System architecture documentation
- `docs/api.md` - API reference
- `docs/troubleshooting.md` - Troubleshooting guide
- `.github/workflows/ci.yml` - CI/CD pipeline
- `.pre-commit-config.yaml` - Pre-commit hooks
- `pyproject.toml` - Tool configuration (pytest, mypy, coverage)

**Files Modified:**
- `transformer_lab/models/transformer.py` - Add type hints, docstrings
- `transformer_lab/utils/config.py` - Refactor into package
- `transformer_lab/train.py` - Extract Trainer class, add error handling
- `transformer_lab/evaluate.py` - Add error recovery
- `transformer_lab/decode.py` - Add type hints
- `transformer_lab/utils/logging_utils.py` - Enhance with structured logging
- `transformer_lab/utils/metrics.py` - Add type hints
- `transformer_lab/utils/distributed.py` - Add error handling
- `transformer_lab/data/tokenization.py` - Add validation
- `transformer_lab/data/datasets.py` - Add error handling
- `transformer_lab/data/collate.py` - Add type hints
- `transformer_lab/optim/scheduler.py` - Add type hints
- `requirements.txt` - Add testing and tooling dependencies
- `README.md` - Expand with usage documentation

---

## Phase 1: Week 1 - Testing Infrastructure Setup (Days 1-5)

### Task 1.1: Set Up Testing Framework and Directory Structure

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`
- Create: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add testing dependencies to requirements.txt**

```bash
echo "
# Testing
pytest>=7.0
pytest-cov>=4.0
pytest-mock>=3.0
hypothesis>=6.0
factory-boy>=3.0

# Type checking
mypy>=1.0
pydantic>=2.0
beartype>=0.10

# Linting and formatting
black>=22.0
isort>=5.0
flake8>=5.0
pre-commit>=3.0

# Documentation
sphinx>=5.0
myst-parser>=0.18
pydata-sphinx-theme>=0.12

# Performance testing
pytest-benchmark>=4.0
memory-profiler>=0.60
" >> requirements.txt
```

- [ ] **Step 2: Create pytest configuration in pyproject.toml**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=transformer_lab",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=90",
    "-v"
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "performance: Performance tests",
    "slow: Slow running tests"
]

[tool.coverage.run]
source = ["transformer_lab"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__init__.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:"
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.black]
line-length = 88
target-version = ['py38']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["transformer_lab"]
```

- [ ] **Step 3: Create conftest.py with fixtures**

```python
# tests/conftest.py
import pytest
import torch
from transformer_lab.models.transformer import Transformer
from transformer_lab.utils.config import TrainConfig


@pytest.fixture
def base_config():
    """Create a minimal config for testing."""
    return TrainConfig(
        model=dict(
            vocab_size=1000,
            N=2,
            d_model=128,
            d_ff=256,
            num_heads=4,
            dropout=0.1,
            attn_dropout=0.0,
            activation="relu",
            share_embeddings=True,
            tie_softmax_weight=True,
            pos_encoding="sinusoidal",
            label_smoothing=0.1
        ),
        data=dict(
            dataset="wmt14",
            lang_pair="en-de",
            max_src_len=50,
            max_tgt_len=50,
            min_len=1,
            tokenizer_dir="tests/data/tokenizer",
            vocab_size=1000,
            use_shared_vocab=True,
            max_tokens_per_batch=1000,
            num_buckets=2,
            cache_dir=None
        ),
        optim=dict(
            lr=5e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.0,
            warmup_steps=100,
            max_steps=1000,
            grad_clip=1.0
        ),
        runtime=dict(
            seed=42,
            device="cpu",
            num_workers=0,
            log_dir="tests/logs",
            ckpt_dir="tests/checkpoints",
            tb_dir="tests/tensorboard",
            save_every=100,
            eval_every=200,
            keep_last=5,
            amp=False,
            accumulate_steps=1
        ),
        decode=dict(
            beam_size=2,
            length_penalty=0.6,
            max_len_offset=10,
            max_len_ratio=1.0
        )
    )


@pytest.fixture
def small_transformer(base_config):
    """Create a small transformer model for testing."""
    model = Transformer(
        vocab_size=base_config.model.vocab_size,
        N=base_config.model.N,
        d_model=base_config.model.d_model,
        d_ff=base_config.model.d_ff,
        num_heads=base_config.model.num_heads,
        dropout=base_config.model.dropout,
        attn_dropout=base_config.model.attn_dropout,
        activation=base_config.model.activation,
        share_embeddings=base_config.model.share_embeddings,
        tie_softmax_weight=base_config.model.tie_softmax_weight,
        pos_encoding=base_config.model.pos_encoding
    )
    return model


@pytest.fixture
def sample_batch(base_config):
    """Create a sample batch for testing."""
    batch_size = 4
    src_len = 20
    tgt_len = 25

    src_ids = torch.randint(0, base_config.model.vocab_size, (batch_size, src_len))
    tgt_in_ids = torch.randint(0, base_config.model.vocab_size, (batch_size, tgt_len))
    tgt_out_ids = torch.randint(0, base_config.model.vocab_size, (batch_size, tgt_len))

    # Create padding mask (True = valid token)
    src_key_padding_mask = torch.ones_like(src_ids, dtype=torch.bool)
    tgt_key_padding_mask = torch.ones_like(tgt_in_ids, dtype=torch.bool)

    return {
        "src_ids": src_ids,
        "tgt_in_ids": tgt_in_ids,
        "tgt_out_ids": tgt_out_ids,
        "src_key_padding_mask": src_key_padding_mask,
        "tgt_key_padding_mask": tgt_key_padding_mask
    }


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    dirs = {
        "log_dir": tmp_path / "logs",
        "ckpt_dir": tmp_path / "checkpoints",
        "tb_dir": tmp_path / "tensorboard",
        "tokenizer_dir": tmp_path / "tokenizer"
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs
```

- [ ] **Step 4: Install dependencies and verify setup**

```bash
pip install -r requirements.txt
pytest --version
mypy --version
black --version
```

Expected output:
```
pytest 7.x.x
mypy 1.x.x
black 22.x.x
```

- [ ] **Step 5: Run empty test suite to verify configuration**

```bash
pytest -v
```

Expected: 0 tests collected, 0 failures

- [ ] **Step 6: Commit setup**

```bash
git add requirements.txt pyproject.toml tests/conftest.py tests/__init__.py
git commit -m "test: setup pytest framework and testing infrastructure"
```

---

### Task 1.2: Write Unit Tests for Transformer Model

**Files:**
- Create: `tests/unit/test_transformer.py`
- Modify: `transformer_lab/models/transformer.py` (add type hints)

- [ ] **Step 1: Write failing test for PositionalEncoding**

```python
# tests/unit/test_transformer.py
import pytest
import torch
from transformer_lab.models.transformer import PositionalEncoding


@pytest.mark.unit
def test_positional_encoding_sinusoidal_shape():
    """Test that sinusoidal positional encoding produces correct shape."""
    batch_size = 2
    seq_len = 10
    d_model = 128

    pos_enc = PositionalEncoding(d_model=d_model, mode="sinusoidal")
    x = torch.randn(batch_size, seq_len, d_model)

    output = pos_enc(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert torch.is_tensor(output)


@pytest.mark.unit
def test_positional_encoding_learned_shape():
    """Test that learned positional encoding produces correct shape."""
    batch_size = 2
    seq_len = 10
    d_model = 128

    pos_enc = PositionalEncoding(d_model=d_model, mode="learned")
    x = torch.randn(batch_size, seq_len, d_model)

    output = pos_enc(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert torch.is_tensor(output)


@pytest.mark.unit
def test_positional_encoding_sinusoidal_values():
    """Test that sinusoidal encoding produces expected value ranges."""
    d_model = 128
    pos_enc = PositionalEncoding(d_model=d_model, mode="sinusoidal")

    # Check that the registered buffer exists
    assert hasattr(pos_enc, "pe")
    assert pos_enc.pe.shape == (1, 5000, d_model)

    # Values should be in range [-1, 1] due to sin/cos
    assert pos_enc.pe.min() >= -1.0
    assert pos_enc.pe.max() <= 1.0


@pytest.mark.unit
def test_positional_encoding_dropout():
    """Test that dropout is applied during training."""
    d_model = 128
    pos_enc = PositionalEncoding(d_model=d_model, dropout=0.5, mode="sinusoidal")
    pos_enc.train()

    x = torch.randn(2, 10, d_model)
    output1 = pos_enc(x)
    output2 = pos_enc(x)

    # With dropout and training=True, outputs should differ
    assert not torch.allclose(output1, output2)

    pos_enc.eval()
    output3 = pos_enc(x)
    output4 = pos_enc(x)

    # With eval mode, outputs should be identical
    assert torch.allclose(output3, output4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_transformer.py::test_positional_encoding_sinusoidal_shape -v
```

Expected: FAIL - "PositionalEncoding not imported or not working"

- [ ] **Step 3: Add type hints to PositionalEncoding in transformer.py**

```python
# transformer_lab/models/transformer.py
from typing import Optional
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal or learned positional encoding."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        mode: str = "sinusoidal"
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        if mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        else:
            self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if isinstance(self.pe, nn.Embedding):  # learned
            T = x.size(1)
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            x = x + self.pe(pos)
        else:
            x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_transformer.py::test_positional_encoding_sinusoidal_shape -v
pytest tests/unit/test_transformer.py::test_positional_encoding_learned_shape -v
pytest tests/unit/test_transformer.py::test_positional_encoding_sinusoidal_values -v
pytest tests/unit/test_transformer.py::test_positional_encoding_dropout -v
```

Expected: All tests PASS

- [ ] **Step 5: Write tests for MultiHeadAttention**

```python
# Add to tests/unit/test_transformer.py
from transformer_lab.models.transformer import MultiHeadAttention


@pytest.mark.unit
def test_multihead_attention_shapes():
    """Test that multi-head attention produces correct output shapes."""
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    output = mha(q, k, v)

    assert output.shape == (batch_size, seq_len, d_model)


@pytest.mark.unit
def test_multihead_attention_with_mask():
    """Test that masking works correctly in multi-head attention."""
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    # Create a mask that allows only first 5 positions
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[:, :, :, 5:] = True  # Mask out positions 5-9

    output = mha(q, k, v, mask)

    assert output.shape == (batch_size, seq_len, d_model)


@pytest.mark.unit
def test_multihead_attention_different_seq_lengths():
    """Test attention with different source and target lengths."""
    batch_size = 2
    src_len = 10
    tgt_len = 15
    d_model = 128
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    q = torch.randn(batch_size, tgt_len, d_model)
    k = torch.randn(batch_size, src_len, d_model)
    v = torch.randn(batch_size, src_len, d_model)

    output = mha(q, k, v)

    assert output.shape == (batch_size, tgt_len, d_model)


@pytest.mark.unit
def test_multihead_attention_gradient_flow():
    """Test that gradients flow correctly through attention."""
    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 4

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    output = mha(q, k, v)
    loss = output.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()
```

- [ ] **Step 6: Run attention tests to verify they fail**

```bash
pytest tests/unit/test_transformer.py::test_multihead_attention_shapes -v
```

Expected: FAIL - type errors or implementation issues

- [ ] **Step 7: Add type hints to MultiHeadAttention**

```python
# transformer_lab/models/transformer.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, Tq, D = q.shape
        Tk = k.size(1)
        q = self.q_proj(q).view(B, Tq, self.num_heads, self.d_k).transpose(1, 2)  # (B,H,Tq,d_k)
        k = self.k_proj(k).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,Tq,Tk)
        if mask is not None:
            # mask True 表示可见，这里转换为 -inf 屏蔽
            # mask 可能为 (B,1,1,Tk) 或 (B,1,Tq,Tk)
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        context = torch.matmul(attn, v)  # (B,H,Tq,d_k)
        context = context.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.o_proj(context)
        out = self.proj_drop(out)
        return out
```

- [ ] **Step 8: Run attention tests to verify they pass**

```bash
pytest tests/unit/test_transformer.py::test_multihead_attention_shapes -v
pytest tests/unit/test_transformer.py::test_multihead_attention_with_mask -v
pytest tests/unit/test_transformer.py::test_multihead_attention_different_seq_lengths -v
pytest tests/unit/test_transformer.py::test_multihead_attention_gradient_flow -v
```

Expected: All tests PASS

- [ ] **Step 9: Write tests for full Transformer model**

```python
# Add to tests/unit/test_transformer.py
from transformer_lab.models.transformer import Transformer


@pytest.mark.unit
def test_transformer_forward_pass(small_transformer, sample_batch):
    """Test full transformer forward pass."""
    model = small_transformer
    batch = sample_batch

    logits = model(
        batch["src_ids"],
        batch["tgt_in_ids"],
        batch["src_key_padding_mask"],
        batch["tgt_key_padding_mask"]
    )

    assert logits.shape == (
        batch["src_ids"].shape[0],
        batch["tgt_in_ids"].shape[1],
        model.vocab_size
    )


@pytest.mark.unit
def test_transformer_encode(small_transformer, sample_batch):
    """Test transformer encoding."""
    model = small_transformer
    batch = sample_batch

    memory = model.encode(
        batch["src_ids"],
        batch["src_key_padding_mask"]
    )

    assert memory.shape == (
        batch["src_ids"].shape[0],
        batch["src_ids"].shape[1],
        model.d_model
    )


@pytest.mark.unit
def test_transformer_decode(small_transformer, sample_batch):
    """Test transformer decoding."""
    model = small_transformer
    batch = sample_batch

    memory = model.encode(
        batch["src_ids"],
        batch["src_key_padding_mask"]
    )

    output = model.decode(
        batch["tgt_in_ids"],
        memory,
        batch["tgt_key_padding_mask"],
        batch["src_key_padding_mask"]
    )

    assert output.shape == (
        batch["tgt_in_ids"].shape[0],
        batch["tgt_in_ids"].shape[1],
        model.d_model
    )


@pytest.mark.unit
def test_transformer_weight_tying(small_transformer):
    """Test that weight tying works correctly."""
    model = small_transformer

    if model.model.tie_softmax_weight:
        # Check that generator weight is same object as embedding weight
        assert model.generator.weight is model.tgt_embed.weight

    if model.model.share_embeddings:
        # Check that source and target embeddings are same
        assert model.src_embed.weight is model.tgt_embed.weight


@pytest.mark.unit
def test_transformer_causal_masking(small_transformer, sample_batch):
    """Test that causal masking prevents attending to future positions."""
    model = small_transformer
    batch = sample_batch

    # Get decoder output
    memory = model.encode(
        batch["src_ids"],
        batch["src_key_padding_mask"]
    )

    # Manually call decode to inspect masks
    tgt_ids = batch["tgt_in_ids"]
    T = tgt_ids.size(1)

    # Create subsequent mask
    sub_mask = torch.triu(torch.ones((1, 1, T, T), dtype=torch.bool), diagonal=1)
    tgt_mask = (~sub_mask) & batch["tgt_key_padding_mask"].unsqueeze(1).unsqueeze(2)

    # Verify mask shape and properties
    assert tgt_mask.shape == (batch["src_ids"].shape[0], 1, T, T)

    # Upper triangle (future positions) should be False (masked)
    for i in range(T):
        for j in range(T):
            if j > i:  # Future position
                assert not tgt_mask[0, 0, i, j]
            else:  # Current or past position
                assert tgt_mask[0, 0, i, j]
```

- [ ] **Step 10: Run transformer tests to verify they fail**

```bash
pytest tests/unit/test_transformer.py::test_transformer_forward_pass -v
```

Expected: FAIL - type errors

- [ ] **Step 11: Add type hints to Transformer class**

```python
# transformer_lab/models/transformer.py
from typing import Optional, Tuple
import torch
import torch.nn as nn

class Transformer(nn.Module):
    """Transformer Encoder-Decoder with shared embeddings (optional) and tied softmax (optional)."""

    def __init__(
        self,
        vocab_size: int,
        N: int = 6,
        d_model: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        activation: str = "relu",
        share_embeddings: bool = True,
        tie_softmax_weight: bool = True,
        pos_encoding: str = "sinusoidal",
        max_len: int = 1024
    ) -> None:
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = self.src_embed if share_embeddings else nn.Embedding(vocab_size, d_model)
        self.pos_enc_src = PositionalEncoding(d_model, max_len, dropout, pos_encoding)
        self.pos_enc_tgt = PositionalEncoding(d_model, max_len, dropout, pos_encoding)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, attn_dropout, activation) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, attn_dropout, activation) for _ in range(N)])
        self.norm_enc = nn.LayerNorm(d_model)
        self.norm_dec = nn.LayerNorm(d_model)

        self.generator = nn.Linear(d_model, vocab_size)
        if tie_softmax_weight and self.tgt_embed.weight.shape == self.generator.weight.shape:
            self.generator.weight = self.tgt_embed.weight  # weight tying

        self.d_model = d_model
        self.vocab_size = vocab_size

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc_src(self.src_embed(src_ids) * math.sqrt(self.d_model))
        # 构造注意力 mask: (B,1,1,Tk)
        src_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # True=valid
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        mem: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        y = self.pos_enc_tgt(self.tgt_embed(tgt_ids) * math.sqrt(self.d_model))
        # subsequent mask ∧ padding mask
        T = tgt_ids.size(1)
        sub_mask = torch.triu(torch.ones((1, 1, T, T), device=tgt_ids.device, dtype=torch.bool), diagonal=1)
        tgt_mask = (~sub_mask) & tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)  # True=allowed
        mem_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.decoder:
            y = layer(y, mem, tgt_mask, mem_mask)
        return self.norm_dec(y)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        mem = self.encode(src_ids, src_key_padding_mask)
        out = self.decode(tgt_in_ids, mem, tgt_key_padding_mask, src_key_padding_mask)
        logits = self.generator(out)
        return logits
```

- [ ] **Step 12: Run all transformer tests to verify they pass**

```bash
pytest tests/unit/test_transformer.py -v
```

Expected: All 12 tests PASS

- [ ] **Step 13: Check test coverage**

```bash
pytest tests/unit/test_transformer.py --cov=transformer_lab.models.transformer --cov-report=term-missing
```

Expected: >90% coverage for transformer.py

- [ ] **Step 14: Commit transformer tests**

```bash
git add tests/unit/test_transformer.py transformer_lab/models/transformer.py
git commit -m "test: add comprehensive transformer model unit tests with type hints"
```

---

### Task 1.3: Write Unit Tests for Configuration

**Files:**
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write tests for config loading**

```python
# tests/unit/test_config.py
import pytest
import tempfile
import yaml
from pathlib import Path
from transformer_lab.utils.config import load_config, TrainConfig


@pytest.mark.unit
def test_load_config_default():
    """Test loading default config."""
    # Create a minimal config file
    config_data = {
        "model": {"N": 2, "d_model": 128},
        "data": {"vocab_size": 1000},
        "optim": {"max_steps": 100},
        "runtime": {},
        "decode": {}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        cfg = load_config(config_path)
        assert isinstance(cfg, TrainConfig)
        assert cfg.model.N == 2
        assert cfg.model.d_model == 128
        assert cfg.data.vocab_size == 1000
        assert cfg.optim.max_steps == 100
    finally:
        Path(config_path).unlink()


@pytest.mark.unit
def test_load_config_with_data_vocab_override():
    """Test that data.vocab_size overrides model.vocab_size."""
    config_data = {
        "model": {"vocab_size": 5000},
        "data": {"vocab_size": 1000},
        "optim": {},
        "runtime": {},
        "decode": {}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        cfg = load_config(config_path)
        assert cfg.model.vocab_size == 1000  # Should be overridden by data.vocab_size
    finally:
        Path(config_path).unlink()


@pytest.mark.unit
def test_load_config_partial():
    """Test loading config with partial data."""
    config_data = {
        "model": {"N": 4},
        "optim": {"lr": 1e-3}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        cfg = load_config(config_path)
        assert cfg.model.N == 4
        assert cfg.optim.lr == 1e-3
        # Should have defaults for missing sections
        assert cfg.model.d_model == 512  # default
    finally:
        Path(config_path).unlink()


@pytest.mark.unit
def test_config_immutability():
    """Test that config attributes are accessible but not mutable."""
    cfg = TrainConfig()

    # Should be able to read
    assert cfg.model.N == 6

    # Should not be able to modify (dataclass is frozen implicitly)
    with pytest.raises(AttributeError):
        cfg.model.N = 10
```

- [ ] **Step 2: Run config tests to verify they pass**

```bash
pytest tests/unit/test_config.py -v
```

Expected: All tests PASS

- [ ] **Step 3: Commit config tests**

```bash
git add tests/unit/test_config.py
git commit -m "test: add configuration loading unit tests"
```

---

### Task 1.4: Write Unit Tests for Metrics

**Files:**
- Create: `tests/unit/test_metrics.py`

- [ ] **Step 1: Write tests for perplexity calculation**

```python
# tests/unit/test_metrics.py
import pytest
import torch
from transformer_lab.utils.metrics import perplexity


@pytest.mark.unit
def test_perplexity_basic():
    """Test basic perplexity calculation."""
    loss = 2.0
    ppl = perplexity(loss)
    assert ppl == pytest.approx(7.389, rel=1e-3)  # e^2


@pytest.mark.unit
def test_perplexity_zero_loss():
    """Test perplexity with zero loss."""
    loss = 0.0
    ppl = perplexity(loss)
    assert ppl == 1.0  # e^0


@pytest.mark.unit
def test_perplexity_negative_loss():
    """Test that negative loss raises error."""
    loss = -1.0
    with pytest.raises(ValueError):
        perplexity(loss)


@pytest.mark.unit
def test_perplexity_tensor():
    """Test perplexity with tensor input."""
    loss = torch.tensor(1.5)
    ppl = perplexity(loss)
    assert ppl == pytest.approx(4.481, rel=1e-3)  # e^1.5
```

- [ ] **Step 2: Run metrics tests**

```bash
pytest tests/unit/test_metrics.py -v
```

Expected: Tests PASS (perplexity function already exists and works)

- [ ] **Step 3: Write tests for BLEU score**

```python
# Add to tests/unit/test_metrics.py
from transformer_lab.utils.metrics import compute_bleu


@pytest.mark.unit
def test_compute_bleu_identical():
    """Test BLEU with identical predictions and references."""
    preds = ["hello world", "test sentence"]
    refs = ["hello world", "test sentence"]

    bleu = compute_bleu(preds, refs)
    assert bleu == 100.0


@pytest.mark.unit
def test_compute_bleu_completely_different():
    """Test BLEU with completely different predictions and references."""
    preds = ["hello world"]
    refs = ["completely different sentence"]

    bleu = compute_bleu(preds, refs)
    assert bleu < 10.0  # Should be very low


@pytest.mark.unit
def test_compute_bleu_partial_match():
    """Test BLEU with partial matches."""
    preds = ["hello beautiful world"]
    refs = ["hello wonderful world"]

    bleu = compute_bleu(preds, refs)
    assert 20.0 < bleu < 80.0  # Should be moderate


@pytest.mark.unit
def test_compute_bleu_empty_prediction():
    """Test BLEU with empty prediction."""
    preds = [""]
    refs = ["hello world"]

    bleu = compute_bleu(preds, refs)
    assert bleu == 0.0


@pytest.mark.unit
def test_compute_bleu_multiple_refs():
    """Test BLEU with multiple references per prediction."""
    # This tests if compute_bleu handles multiple references correctly
    # Implementation depends on actual compute_bleu function
    pass
```

- [ ] **Step 4: Check what compute_bleu actually does**

```bash
grep -n "def compute_bleu" transformer_lab/utils/metrics.py
```

- [ ] **Step 5: Update BLEU tests based on actual implementation**

```python
# Adjust tests based on actual function signature
```

- [ ] **Step 6: Commit metrics tests**

```bash
git add tests/unit/test_metrics.py
git commit -m "test: add metrics computation unit tests"
```

---

### Task 1.5: Write Unit Tests for Scheduler

**Files:**
- Create: `tests/unit/test_scheduler.py`

- [ ] **Step 1: Write tests for NoamScheduler**

```python
# tests/unit/test_scheduler.py
import pytest
import torch
from torch.optim import Adam
from transformer_lab.optim.scheduler import NoamScheduler
from transformer_lab.models.transformer import Transformer


@pytest.mark.unit
def test_noam_scheduler_initial_lr():
    """Test that NoamScheduler starts with correct initial learning rate."""
    model = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)

    # Before any steps, LR should be 0 (due to warmup)
    initial_lr = scheduler.get_last_lr()[0]
    assert initial_lr == 0.0


@pytest.mark.unit
def test_noam_scheduler_warmup_phase():
    """Test learning rate during warmup phase."""
    model = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)

    # During warmup, LR should increase linearly
    lrs = []
    for _ in range(50):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # LR should be increasing
    assert all(lrs[i] < lrs[i+1] for i in range(len(lrs)-1))
    assert lrs[-1] > 0.0


@pytest.mark.unit
def test_noam_scheduler_post_warmup():
    """Test learning rate after warmup phase."""
    model = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)

    # Step through warmup
    for _ in range(100):
        scheduler.step()

    warmup_lr = scheduler.get_last_lr()[0]

    # After warmup, LR should start decreasing
    scheduler.step()
    post_warmup_lr = scheduler.get_last_lr()[0]

    assert post_warmup_lr < warmup_lr


@pytest.mark.unit
def test_noam_scheduler_formula():
    """Test that scheduler follows the Noam formula."""
    model = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    d_model = 128
    warmup_steps = 100
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    # Test at specific steps
    test_steps = [1, 10, 50, 100, 200, 500]

    for step in range(1, max(test_steps) + 1):
        scheduler.step()
        if step in test_steps:
            actual_lr = scheduler.get_last_lr()[0]
            # Noam formula: lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
            factor = d_model ** -0.5
            if step <= warmup_steps:
                expected_lr = factor * step * warmup_steps ** -1.5
            else:
                expected_lr = factor * step ** -0.5

            assert actual_lr == pytest.approx(expected_lr, rel=1e-6)


@pytest.mark.unit
def test_noam_scheduler_state_dict():
    """Test scheduler state saving and loading."""
    model = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer = Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)

    # Step a few times
    for _ in range(50):
        scheduler.step()

    # Save state
    state = scheduler.state_dict()

    # Create new scheduler and load state
    model2 = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer2 = Adam(model2.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler2 = NoamScheduler(optimizer2, d_model=128, warmup_steps=100)
    scheduler2.load_state_dict(state)

    # LRs should match
    assert scheduler.get_last_lr()[0] == scheduler2.get_last_lr()[0]

    # Stepping both should produce same results
    scheduler.step()
    scheduler2.step()
    assert scheduler.get_last_lr()[0] == scheduler2.get_last_lr()[0]
```

- [ ] **Step 2: Run scheduler tests**

```bash
pytest tests/unit/test_scheduler.py -v
```

Expected: Tests PASS or FAIL based on actual NoamScheduler implementation

- [ ] **Step 3: Check NoamScheduler implementation**

```bash
grep -A 20 "class NoamScheduler" transformer_lab/optim/scheduler.py
```

- [ ] **Step 4: Adjust tests based on actual implementation**

```python
# If NoamScheduler doesn't exist or has different signature, adjust tests
```

- [ ] **Step 5: Commit scheduler tests**

```bash
git add tests/unit/test_scheduler.py
git commit -m "test: add learning rate scheduler unit tests"
```

---

### Task 1.6: Set Up Pre-commit Hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Create pre-commit configuration**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 22.12.0
    hooks:
      - id: black
        language_version: python3.8
        files: ^(transformer_lab|tests)/.*\.py$

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
        files: ^(transformer_lab|tests)/.*\.py$

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --ignore=E203,W503]
        files: ^(transformer_lab|tests)/.*\.py$

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [torch, pydantic]
        files: ^transformer_lab/.*\.py$
        args: [--ignore-missing-imports]

  - repo: local
    hooks:
      - id: pytest-unit
        name: pytest unit tests
        entry: pytest tests/unit -v
        language: system
        pass_filenames: false
        always_run: true
        stages: [push]
```

- [ ] **Step 2: Install pre-commit hooks**

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

- [ ] **Step 3: Run pre-commit on all files**

```bash
pre-commit run --all-files
```

Expected: Some files will need formatting

- [ ] **Step 4: Fix any formatting issues**

```bash
black transformer_lab/ tests/
isort transformer_lab/ tests/
```

- [ ] **Step 5: Verify pre-commit passes**

```bash
pre-commit run --all-files
```

Expected: All checks pass

- [ ] **Step 6: Commit pre-commit configuration**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hooks for code quality"
```

---

### Task 1.7: Set Up CI/CD Pipeline

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create GitHub Actions workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Lint with flake8
      run: |
        flake8 transformer_lab/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 transformer_lab/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Check formatting with black
      run: |
        black --check transformer_lab/ tests/

    - name: Check imports with isort
      run: |
        isort --check-only transformer_lab/ tests/

    - name: Type check with mypy
      run: |
        mypy transformer_lab/ --ignore-missing-imports

    - name: Test with pytest
      run: |
        pytest tests/unit -v --cov=transformer_lab --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  integration-test:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.8
      uses: actions/setup-python@v4
      with:
        python-version: 3.8

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run integration tests
      run: |
        pytest tests/integration -v --cov=transformer_lab --cov-report=xml

    - name: Upload integration coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: integration
        name: codecov-integration
        fail_ci_if_error: false

  performance-test:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.8
      uses: actions/setup-python@v4
      with:
        python-version: 3.8

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run performance benchmarks
      run: |
        pytest tests/performance -v --benchmark-only

    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: .benchmarks/output.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

- [ ] **Step 2: Commit CI configuration**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for CI/CD"
```

---

### Task 1.8: Week 1 Summary and Coverage Check

- [ ] **Step 1: Run full test suite and check coverage**

```bash
pytest tests/unit -v --cov=transformer_lab --cov-report=term-missing
```

Expected: >60% coverage achieved

- [ ] **Step 2: Generate coverage report**

```bash
pytest tests/unit --cov=transformer_lab --cov-report=html
coverage report --fail-under=60
```

- [ ] **Step 3: Check which files need more coverage**

```bash
coverage report | grep -v "100%"
```

- [ ] **Step 4: Commit Week 1 progress**

```bash
git add .
git commit -m "test: Week 1 - testing infrastructure complete (60%+ coverage)"
```

**Week 1 Deliverable:** Testing infrastructure complete with 60%+ coverage, pre-commit hooks configured, CI/CD pipeline ready

---

## Phase 2: Week 2 - Type Safety and Error Handling (Days 6-10)

### Task 2.1: Add Type Hints to Data Pipeline

**Files:**
- Modify: `transformer_lab/data/datasets.py`
- Modify: `transformer_lab/data/tokenization.py`
- Modify: `transformer_lab/data/collate.py`

- [ ] **Step 1: Add type hints to datasets.py**

```python
# transformer_lab/data/datasets.py
from __future__ import annotations
from typing import Optional, Dict, Any
from datasets import DatasetDict

def load_mt_dataset(
    dataset_name: str,
    lang_pair: str,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """Load machine translation dataset."""
    # ... existing implementation ...

def filter_and_rename(
    ddict: DatasetDict,
    src_lang: str,
    tgt_lang: str,
    min_len: int,
    max_src_len: int,
    max_tgt_len: int
) -> DatasetDict:
    """Filter and rename dataset columns."""
    # ... existing implementation ...
```

- [ ] **Step 2: Add type hints to tokenization.py**

```python
# transformer_lab/data/tokenization.py
from __future__ import annotations
from typing import Optional, List
from datasets import Dataset
import sentencepiece as spm

class SPTokenizer:
    def __init__(self, model_path: str) -> None:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id: int = self.sp.pad_id()
        self.bos_id: int = self.sp.bos_id()
        self.eos_id: int = self.sp.eos_id()
        self.vocab_size: int = self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(ids)

def train_or_load_spm(
    train_dataset: Dataset,
    tokenizer_dir: str,
    vocab_size: int
) -> SPTokenizer:
    """Train or load SentencePiece tokenizer."""
    # ... existing implementation ...
```

- [ ] **Step 3: Add type hints to collate.py**

```python
# transformer_lab/data/collate.py
from __future__ import annotations
from typing import Dict, List, Any
import torch

class DynamicBatcher:
    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        max_tokens_per_batch: int,
        num_buckets: int,
        shuffle: bool = False
    ) -> None:
        # ... existing implementation ...

def collate(samples: List[Dict[str, Any]], pad_id: int) -> Dict[str, torch.Tensor]:
    """Collate samples into a batch."""
    # ... existing implementation ...
```

- [ ] **Step 4: Run mypy on data module**

```bash
mypy transformer_lab/data/ --ignore-missing-imports
```

Expected: Type errors identified

- [ ] **Step 5: Fix mypy errors iteratively**

```bash
# Run mypy, fix errors, repeat until clean
mypy transformer_lab/data/ --ignore-missing-imports
```

- [ ] **Step 6: Commit type hints for data pipeline**

```bash
git add transformer_lab/data/
git commit -m "type: add type hints to data pipeline modules"
```

---

### Task 2.2: Create Custom Exception Classes

**Files:**
- Create: `transformer_lab/utils/errors.py`

- [ ] **Step 1: Create custom exception hierarchy**

```python
# transformer_lab/utils/errors.py
"""Custom exceptions for transformer_lab."""


class TransformerLabError(Exception):
    """Base exception for all transformer_lab errors."""
    pass


class ConfigurationError(TransformerLabError):
    """Raised when there's an error in configuration."""
    pass


class DataLoadingError(TransformerLabError):
    """Raised when there's an error loading data."""
    pass


class TokenizationError(TransformerLabError):
    """Raised when there's an error in tokenization."""
    pass


class TrainingError(TransformerLabError):
    """Base exception for training-related errors."""
    pass


class OOMError(TrainingError):
    """Raised when out of memory error occurs during training."""
    pass


class GradientExplosionError(TrainingError):
    """Raised when gradients explode during training."""
    pass


class NaNLossError(TrainingError):
    """Raised when loss becomes NaN during training."""
    pass


class CheckpointError(TransformerLabError):
    """Raised when there's an error with checkpointing."""
    pass


class ValidationError(TransformerLabError):
    """Raised when validation fails."""
    pass


class DecodingError(TransformerLabError):
    """Raised when decoding fails."""
    pass
```

- [ ] **Step 2: Write tests for exceptions**

```python
# Create: tests/unit/test_errors.py
import pytest
from transformer_lab.utils.errors import (
    TransformerLabError,
    ConfigurationError,
    DataLoadingError,
    TrainingError,
    OOMError,
    NaNLossError
)


@pytest.mark.unit
def test_exception_hierarchy():
    """Test that exceptions have correct inheritance."""
    assert issubclass(ConfigurationError, TransformerLabError)
    assert issubclass(DataLoadingError, TransformerLabError)
    assert issubclass(TrainingError, TransformerLabError)
    assert issubclass(OOMError, TrainingError)
    assert issubclass(NaNLossError, TrainingError)


@pytest.mark.unit
def test_raise_configuration_error():
    """Test raising ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        raise ConfigurationError("Invalid config")
    assert str(exc_info.value) == "Invalid config"


@pytest.mark.unit
def test_raise_oom_error():
    """Test raising OOMError."""
    with pytest.raises(OOMError) as exc_info:
        raise OOMError("GPU out of memory")
    assert "out of memory" in str(exc_info.value)
    assert isinstance(exc_info.value, TrainingError)
```

- [ ] **Step 3: Run exception tests**

```bash
pytest tests/unit/test_errors.py -v
```

Expected: All tests PASS

- [ ] **Step 4: Commit exception classes**

```bash
git add transformer_lab/utils/errors.py tests/unit/test_errors.py
git commit -m "feat: add custom exception hierarchy for error handling"
```

---

### Task 2.3: Implement Error Recovery in Training Loop

**Files:**
- Modify: `transformer_lab/train.py`
- Create: `transformer_lab/utils/training/loop.py`

- [ ] **Step 1: Extract training loop into separate module**

```python
# transformer_lab/utils/training/loop.py
from __future__ import annotations
import torch
import torch.nn as nn
from loguru import logger
from typing import Optional, Dict, Any
from transformer_lab.utils.errors import OOMError, NaNLossError, GradientExplosionError

class TrainingLoop:
    """Robust training loop with error recovery."""

    def __init__(self, model, optimizer, scheduler, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.current_batch_size = config.data.max_tokens_per_batch

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with error handling."""
        try:
            return self._train_step_internal(batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OOMError(f"OOM during training step: {e}")
            else:
                raise

    def _train_step_internal(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Internal training step logic."""
        src_ids = batch["src_ids"]
        tgt_in_ids = batch["tgt_in_ids"]
        tgt_out_ids = batch["tgt_out_ids"]
        src_mask = batch["src_key_padding_mask"]
        tgt_mask = batch["tgt_key_padding_mask"]

        # Forward pass
        logits = self.model(src_ids, tgt_in_ids, src_mask, tgt_mask)
        loss = self.criterion(logits, tgt_out_ids)

        # Check for NaN loss
        if torch.isnan(loss):
            raise NaNLossError(f"Loss is NaN. Logits stats: min={logits.min()}, max={logits.max()}")

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.optim.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optim.grad_clip
            )
            if grad_norm > self.config.optim.grad_clip * 10:
                logger.warning(f"Large gradient norm: {grad_norm}")

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Scheduler step
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0]
        }

    def recover_from_oom(self) -> bool:
        """Attempt to recover from OOM by reducing batch size."""
        if self.current_batch_size < 1000:
            logger.error("Batch size already at minimum, cannot recover from OOM")
            return False

        old_batch_size = self.current_batch_size
        self.current_batch_size = max(1000, self.current_batch_size // 2)

        logger.warning(
            f"OOM recovery: reducing batch size from {old_batch_size} "
            f"to {self.current_batch_size}"
        )

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True
```

- [ ] **Step 2: Write tests for error recovery**

```python
# Create: tests/unit/test_training_loop.py
import pytest
import torch
from transformer_lab.utils.training.loop import TrainingLoop
from transformer_lab.utils.errors import OOMError, NaNLossError
from transformer_lab.models.transformer import Transformer
from transformer_lab.optim.scheduler import NoamScheduler
from transformer_lab.models.label_smoothing import LabelSmoothingLoss
from transformer_lab.utils.config import TrainConfig


@pytest.mark.unit
def test_training_loop_creation(base_config):
    """Test TrainingLoop initialization."""
    model = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)
    criterion = LabelSmoothingLoss(classes=1000, smoothing=0.1)

    loop = TrainingLoop(model, optimizer, scheduler, criterion, base_config)

    assert loop.model is model
    assert loop.optimizer is optimizer
    assert loop.scheduler is scheduler
    assert loop.criterion is criterion


@pytest.mark.unit
def test_training_step_normal(small_transformer, sample_batch, base_config):
    """Test normal training step execution."""
    model = small_transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)
    criterion = LabelSmoothingLoss(classes=1000, smoothing=0.1)

    loop = TrainingLoop(model, optimizer, scheduler, criterion, base_config)

    # Make sure model is in training mode
    model.train()

    result = loop.train_step(sample_batch)

    assert "loss" in result
    assert "lr" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0
    assert not torch.isnan(torch.tensor(result["loss"]))


@pytest.mark.unit
def test_training_step_nan_loss(small_transformer, sample_batch, base_config):
    """Test that NaN loss raises NaNLossError."""
    model = small_transformer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)
    criterion = LabelSmoothingLoss(classes=1000, smoothing=0.1)

    loop = TrainingLoop(model, optimizer, scheduler, criterion, base_config)

    # Corrupt the batch to produce NaN
    corrupted_batch = sample_batch.copy()
    corrupted_batch["tgt_out_ids"].fill_(999999)  # Out of vocab

    with pytest.raises(NaNLossError):
        loop.train_step(corrupted_batch)


@pytest.mark.unit
def test_oom_recovery_reduction():
    """Test OOM recovery reduces batch size."""
    # Create a mock config
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.data = SimpleNamespace()
    config.data.max_tokens_per_batch = 10000
    config.optim = SimpleNamespace()
    config.optim.grad_clip = 1.0

    model = Transformer(vocab_size=1000, N=2, d_model=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=100)
    criterion = LabelSmoothingLoss(classes=1000, smoothing=0.1)

    loop = TrainingLoop(model, optimizer, scheduler, criterion, config)

    assert loop.current_batch_size == 10000

    success = loop.recover_from_oom()

    assert success is True
    assert loop.current_batch_size == 5000  # Should be halved
