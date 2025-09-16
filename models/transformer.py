# transformer_lab/models/transformer.py
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal or learned positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1, mode: str = "sinusoidal"):
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
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

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
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

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, attn_dropout, activation):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.drop(attn_out))
        # FFN + residual + norm
        ff_out = self.ffn(x)
        x = self.norm2(x + self.drop(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, attn_dropout, activation):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, attn_dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask, mem_mask):
        # masked self-attn
        sa = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop(sa))
        # cross-attn
        ca = self.cross_attn(x, mem, mem, mem_mask)
        x = self.norm2(x + self.drop(ca))
        # ffn
        ff = self.ffn(x)
        x = self.norm3(x + self.drop(ff))
        return x

class Transformer(nn.Module):
    """Transformer Encoder-Decoder with shared embeddings (optional) and tied softmax (optional)."""
    def __init__(self, vocab_size: int, N: int = 6, d_model: int = 512, d_ff: int = 2048,
                 num_heads: int = 8, dropout: float = 0.1, attn_dropout: float = 0.0,
                 activation: str = "relu", share_embeddings: bool = True, tie_softmax_weight: bool = True,
                 pos_encoding: str = "sinusoidal", max_len: int = 1024):
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

    def encode(self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor):
        x = self.pos_enc_src(self.src_embed(src_ids) * math.sqrt(self.d_model))
        # 构造注意力 mask: (B,1,1,Tk)
        src_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # True=valid
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(self, tgt_ids: torch.Tensor, mem: torch.Tensor,
               tgt_key_padding_mask: torch.Tensor, src_key_padding_mask: torch.Tensor):
        y = self.pos_enc_tgt(self.tgt_embed(tgt_ids) * math.sqrt(self.d_model))
        # subsequent mask ∧ padding mask
        T = tgt_ids.size(1)
        sub_mask = torch.triu(torch.ones((1, 1, T, T), device=tgt_ids.device, dtype=torch.bool), diagonal=1)
        tgt_mask = (~sub_mask) & tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)  # True=allowed
        mem_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.decoder:
            y = layer(y, mem, tgt_mask, mem_mask)
        return self.norm_dec(y)

    def forward(self, src_ids, tgt_in_ids, src_key_padding_mask, tgt_key_padding_mask):
        mem = self.encode(src_ids, src_key_padding_mask)
        out = self.decode(tgt_in_ids, mem, tgt_key_padding_mask, src_key_padding_mask)
        logits = self.generator(out)
        return logits
