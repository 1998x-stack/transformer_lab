import tempfile
from pathlib import Path

import pytest

from utils.config import TrainConfig, load_config


class TestConfigLoading:
    def test_load_config_default(self):
        config_content = """
model:
  N: 6
  d_model: 512
  d_ff: 2048
  num_heads: 8
  dropout: 0.1
  vocab_size: 37000

data:
  dataset: wmt14
  lang_pair: en-de
  max_src_len: 256
  max_tgt_len: 256
  vocab_size: 37000

optim:
  lr: 0.0005
  warmup_steps: 4000
  max_steps: 100000

runtime:
  seed: 42
  device: cuda
  num_workers: 4

decode:
  beam_size: 4
  length_penalty: 0.6
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert isinstance(config, TrainConfig)
            assert config.model.N == 6
            assert config.model.d_model == 512
            assert config.model.vocab_size == 37000
            assert config.data.dataset == "wmt14"
            assert config.data.lang_pair == "en-de"
            assert config.optim.lr == 0.0005
            assert config.runtime.seed == 42
            assert config.decode.beam_size == 4
        finally:
            Path(config_path).unlink()

    def test_load_config_with_data_vocab_override(self):
        config_content = """
model:
  N: 2
  d_model: 64
  vocab_size: 1000

data:
  vocab_size: 5000
  dataset: wmt14
  lang_pair: en-de
  max_src_len: 256
  max_tgt_len: 256

optim:
  lr: 0.001

runtime:
  seed: 42
  device: cpu

decode:
  beam_size: 4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config.model.vocab_size == 5000
            assert config.data.vocab_size == 5000
            assert config.model.N == 2
            assert config.model.d_model == 64
        finally:
            Path(config_path).unlink()

    def test_load_config_partial(self):
        config_content = """
model:
  N: 4
  d_model: 128

data:
  dataset: wmt16
  lang_pair: en-fr

optim:
  lr: 0.002
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert config.model.N == 4
            assert config.model.d_model == 128
            assert config.data.dataset == "wmt16"
            assert config.data.lang_pair == "en-fr"
            assert config.optim.lr == 0.002
            assert config.model.num_heads == 8
            assert config.model.dropout == 0.1
            assert config.runtime.seed == 42
            assert config.runtime.device == "cuda"
        finally:
            Path(config_path).unlink()

    def test_config_immutability(self):
        config_content = """
model:
  N: 2
  d_model: 64
  vocab_size: 1000

data:
  dataset: wmt14
  lang_pair: en-de
  vocab_size: 1000

optim:
  lr: 0.001

runtime:
  seed: 42
  device: cpu

decode:
  beam_size: 4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_config(config_path)

            with pytest.raises(AttributeError):
                config.model.N = 10

            with pytest.raises(AttributeError):
                config.data.dataset = "wmt16"

            with pytest.raises(AttributeError):
                config.optim.lr = 0.01

            with pytest.raises(AttributeError):
                config.runtime.seed = 100

            with pytest.raises(AttributeError):
                config.decode.beam_size = 8
        finally:
            Path(config_path).unlink()
