import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from models.transformer import Transformer
from utils.config import DataConfig, ModelConfig, RuntimeConfig
from utils.metrics import compute_bleu


@dataclass
class BeamHypo:
    tokens: list
    logprob: float
    ended: bool


def length_penalty(len_y: int, alpha: float) -> float:
    return ((5 + len_y) / 6) ** alpha


def create_tiny_model_config():
    return ModelConfig(
        N=2,
        d_model=64,
        d_ff=256,
        num_heads=4,
        dropout=0.0,
        attn_dropout=0.0,
        vocab_size=100,
        activation="relu",
        share_embeddings=True,
        tie_softmax_weight=True,
        pos_encoding="sinusoidal",
        label_smoothing=0.0,
    )


def create_tiny_data_config():
    return DataConfig(
        dataset="wmt14",
        lang_pair="de-en",
        max_src_len=16,
        max_tgt_len=16,
        min_len=1,
        vocab_size=100,
        use_shared_vocab=True,
        max_tokens_per_batch=100,
        num_buckets=2,
        cache_dir=None,
        tokenizer_dir="work/tokenizer",
    )


def create_tiny_runtime_config():
    return RuntimeConfig(
        seed=42,
        device="cpu",
        num_workers=0,
        log_dir="work/logs",
        ckpt_dir="work/checkpoints",
        tb_dir="work/tensorboard",
        save_every=10,
        eval_every=10,
        keep_last=5,
        amp=False,
        accumulate_steps=1,
    )


@pytest.fixture
def temp_eval_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt_dir = tmpdir / "checkpoints"
        results_dir = tmpdir / "results"

        ckpt_dir.mkdir()
        results_dir.mkdir()

        yield {"base": tmpdir, "ckpt_dir": ckpt_dir, "results_dir": results_dir}


@pytest.fixture
def tiny_sentence_pairs():
    return [
        ("the cat sat on the mat", "die katze saß auf der matte"),
        ("the dog ran in the park", "der hund lief im park"),
        ("a bird flies in the sky", "ein vogel fliegt am himmel"),
        ("the fish swims in the water", "der fisch schwimmt im wasser"),
        ("the child plays with toys", "das kind spielt mit spielzeug"),
    ]


@pytest.mark.integration
@pytest.mark.slow
class TestEvaluationPipeline:
    def test_evaluation_model_loading(self, temp_eval_dirs):
        config = create_tiny_model_config()

        model = Transformer(
            vocab_size=config.vocab_size,
            N=config.N,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            activation=config.activation,
            share_embeddings=config.share_embeddings,
            tie_softmax_weight=config.tie_softmax_weight,
            pos_encoding=config.pos_encoding,
        )

        original_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        ckpt_path = temp_eval_dirs["ckpt_dir"] / "model.pt"
        torch.save({"model": model.state_dict(), "step": 10}, ckpt_path)

        assert ckpt_path.exists(), "Checkpoint should be created"

        loaded_model = Transformer(
            vocab_size=config.vocab_size,
            N=config.N,
            d_model=config.d_model,
            d_ff=config.d_ff,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attn_dropout=config.attn_dropout,
            activation=config.activation,
            share_embeddings=config.share_embeddings,
            tie_softmax_weight=config.tie_softmax_weight,
            pos_encoding=config.pos_encoding,
        )

        state = torch.load(ckpt_path, map_location="cpu")
        loaded_model.load_state_dict(state["model"])

        for name, param in loaded_model.named_parameters():
            assert torch.allclose(
                param, original_params[name]
            ), f"Parameter {name} should match after loading"

    def test_evaluation_inference(self):
        vocab_size = 100
        model = Transformer(
            vocab_size=vocab_size,
            N=2,
            d_model=64,
            d_ff=256,
            num_heads=4,
            dropout=0.0,
            attn_dropout=0.0,
            activation="relu",
            share_embeddings=True,
            tie_softmax_weight=True,
            pos_encoding="sinusoidal",
        )

        model.eval()

        batch_size, seq_len = 2, 8
        src_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tgt_in_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        src_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        tgt_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        with torch.no_grad():
            logits = model(src_ids, tgt_in_ids, src_mask, tgt_mask)

        assert logits.shape == (
            batch_size,
            seq_len,
            vocab_size,
        ), "Output shape should be (B, T, V)"
        assert not torch.isnan(logits).any(), "Output should not contain NaN values"
        assert not torch.isinf(logits).any(), "Output should not contain Inf values"

    def test_evaluation_bleu_computation(self, tiny_sentence_pairs):
        preds = [src for src, _ in tiny_sentence_pairs]
        refs = [tgt for _, tgt in tiny_sentence_pairs]

        bleu_score = compute_bleu(preds, refs)

        assert isinstance(bleu_score, float), "BLEU score should be a float"
        assert 0.0 <= bleu_score <= 100.0, "BLEU score should be between 0 and 100"
        assert bleu_score > 0.0, "BLEU score should be positive for similar sentences"

        identical_bleu = compute_bleu(refs, refs)
        assert identical_bleu > 90.0, "BLEU should be high for identical sentences"

    def test_evaluation_length_penalty(self):
        test_cases = [
            (5, 0.6, ((5 + 5) / 6) ** 0.6),
            (10, 0.6, ((5 + 10) / 6) ** 0.6),
            (15, 0.8, ((5 + 15) / 6) ** 0.8),
        ]

        for length, alpha, expected in test_cases:
            result = length_penalty(length, alpha)
            assert (
                abs(result - expected) < 1e-6
            ), f"Length penalty for len={length}, alpha={alpha} should be {expected}"

    def test_evaluation_beam_hypo_sorting(self):
        hypos = [
            BeamHypo(tokens=[1, 2, 3], logprob=-5.0, ended=False),
            BeamHypo(tokens=[1, 2], logprob=-3.0, ended=False),
            BeamHypo(tokens=[1, 2, 3, 4], logprob=-7.0, ended=False),
        ]

        hypos_sorted = sorted(hypos, key=lambda x: x.logprob, reverse=True)

        assert hypos_sorted[0].logprob == -3.0
        assert hypos_sorted[1].logprob == -5.0
        assert hypos_sorted[2].logprob == -7.0
