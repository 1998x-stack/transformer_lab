import pytest
import torch
import torch.nn as nn
import math
from models.transformer import PositionalEncoding, MultiHeadAttention, Transformer


class TestPositionalEncoding:
    def test_positional_encoding_sinusoidal_shape(self):
        batch_size, seq_len, d_model = 2, 10, 64
        pe = PositionalEncoding(
            d_model, max_len=seq_len, dropout=0.0, mode="sinusoidal"
        )
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_learned_shape(self):
        batch_size, seq_len, d_model = 2, 10, 64
        pe = PositionalEncoding(d_model, max_len=seq_len, dropout=0.0, mode="learned")
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_positional_encoding_sinusoidal_values(self):
        batch_size, seq_len, d_model = 1, 5, 8
        pe = PositionalEncoding(
            d_model, max_len=seq_len, dropout=0.0, mode="sinusoidal"
        )
        x = torch.zeros(batch_size, seq_len, d_model)
        output = pe(x)
        assert not torch.allclose(output, x)
        assert output.min() >= -1.0 and output.max() <= 1.0

    def test_positional_encoding_dropout(self):
        batch_size, seq_len, d_model = 2, 10, 64
        pe = PositionalEncoding(
            d_model, max_len=seq_len, dropout=0.5, mode="sinusoidal"
        )
        pe.train()
        x = torch.randn(batch_size, seq_len, d_model)
        output1 = pe(x)
        output2 = pe(x)
        assert not torch.allclose(output1, output2)


class TestMultiHeadAttention:
    def test_multihead_attention_shapes(self):
        batch_size, seq_len, d_model, num_heads = 2, 10, 64, 4
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        output = mha(q, k, v)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_multihead_attention_with_mask(self):
        batch_size, seq_len, d_model, num_heads = 2, 10, 64, 4
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        q = torch.randn(batch_size, seq_len, d_model)
        k = torch.randn(batch_size, seq_len, d_model)
        v = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
        mask[:, :, :, 5:] = False
        output = mha(q, k, v, mask)
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    def test_multihead_attention_different_seq_lengths(self):
        batch_size, q_len, k_len, d_model, num_heads = 2, 5, 10, 64, 4
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        q = torch.randn(batch_size, q_len, d_model)
        k = torch.randn(batch_size, k_len, d_model)
        v = torch.randn(batch_size, k_len, d_model)
        output = mha(q, k, v)
        assert output.shape == (batch_size, q_len, d_model)

    def test_multihead_attention_gradient_flow(self):
        batch_size, seq_len, d_model, num_heads = 2, 10, 64, 4
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)
        q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output = mha(q, k, v)
        loss = output.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert mha.q_proj.weight.grad is not None


class TestTransformer:
    def test_transformer_forward_pass(self, small_transformer, sample_batch):
        model = small_transformer
        src_ids = sample_batch["src_ids"]
        tgt_in_ids = sample_batch["tgt_in_ids"]
        src_key_padding_mask = sample_batch["src_key_padding_mask"]
        tgt_key_padding_mask = sample_batch["tgt_key_padding_mask"]

        logits = model(src_ids, tgt_in_ids, src_key_padding_mask, tgt_key_padding_mask)
        assert logits.shape == (src_ids.shape[0], tgt_in_ids.shape[1], model.vocab_size)
        assert not torch.isnan(logits).any()

    def test_transformer_encode(self, small_transformer, sample_batch):
        model = small_transformer
        src_ids = sample_batch["src_ids"]
        src_key_padding_mask = sample_batch["src_key_padding_mask"]

        memory = model.encode(src_ids, src_key_padding_mask)
        assert memory.shape == (src_ids.shape[0], src_ids.shape[1], model.d_model)
        assert not torch.isnan(memory).any()

    def test_transformer_decode(self, small_transformer, sample_batch):
        model = small_transformer
        src_ids = sample_batch["src_ids"]
        tgt_in_ids = sample_batch["tgt_in_ids"]
        src_key_padding_mask = sample_batch["src_key_padding_mask"]
        tgt_key_padding_mask = sample_batch["tgt_key_padding_mask"]

        memory = model.encode(src_ids, src_key_padding_mask)
        output = model.decode(
            tgt_in_ids, memory, tgt_key_padding_mask, src_key_padding_mask
        )
        assert output.shape == (tgt_in_ids.shape[0], tgt_in_ids.shape[1], model.d_model)
        assert not torch.isnan(output).any()

    def test_transformer_weight_tying(self, base_config):
        vocab_size, d_model, N = 1000, 64, 2
        model = Transformer(
            vocab_size=vocab_size,
            N=N,
            d_model=d_model,
            d_ff=256,
            num_heads=4,
            share_embeddings=True,
            tie_softmax_weight=True,
        )
        assert model.generator.weight is model.tgt_embed.weight

    def test_transformer_causal_masking(self, small_transformer, sample_batch):
        model = small_transformer
        src_ids = sample_batch["src_ids"]
        tgt_in_ids = sample_batch["tgt_in_ids"]
        src_key_padding_mask = sample_batch["src_key_padding_mask"]
        tgt_key_padding_mask = sample_batch["tgt_key_padding_mask"]

        model.eval()
        with torch.no_grad():
            logits = model(
                src_ids, tgt_in_ids, src_key_padding_mask, tgt_key_padding_mask
            )
        assert logits.shape == (src_ids.shape[0], tgt_in_ids.shape[1], model.vocab_size)
        assert not torch.isnan(logits).any()
