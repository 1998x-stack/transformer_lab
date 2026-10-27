# transformer_lab/models/transformer.py
from __future__ import annotations
import math
from typing import Optional, Tuple
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal or learned positional encoding."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        mode: Literal["sinusoidal", "learned"] = "sinusoidal",
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        if mode == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        elif mode == "learned":
            self.pe = nn.Embedding(max_len, d_model)
        elif mode == "none":
            self.pe = None
        else:
            raise ValueError(f"Unknown positional encoding: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if isinstance(self.pe, nn.Embedding):  # learned
            T = x.size(1)
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            x = x + self.pe(pos)
        elif self.pe is not None:
            if x.size(1) > self.pe.size(1):
                raise ValueError("sequence exceeds positional encoding max_len")
            x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Paper Eq. (1): separately configurable d_k and d_v.

    Masks are boolean with True meaning *visible*. Unscaled logits are a
    mechanism ablation, not the paper's baseline.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 d_k: Optional[int] = None, d_v: Optional[int] = None,
                 attention_scale: bool = True):
        super().__init__()
        if num_heads < 1 or d_model < 1:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads and (d_k is None or d_v is None):
            raise ValueError("Specify both d_k and d_v when d_model is not divisible by heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        if self.d_k < 1 or self.d_v < 1:
            raise ValueError("d_k and d_v must be positive")
        self.attention_scale = attention_scale
        self.q_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.k_proj = nn.Linear(d_model, num_heads * self.d_k)
        self.v_proj = nn.Linear(d_model, num_heads * self.d_v)
        self.o_proj = nn.Linear(num_heads * self.d_v, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.capture_attention = False
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, t_query, _ = q.shape
        t_key = k.size(1)
        def split(tensor: torch.Tensor, width: int) -> torch.Tensor:
            return tensor.reshape(batch, -1, self.num_heads, width).transpose(1, 2)
        queries = split(self.q_proj(q), self.d_k)
        keys = split(self.k_proj(k), self.d_k)
        values = split(self.v_proj(v), self.d_v)
        scores = queries @ keys.transpose(-2, -1)
        if self.attention_scale:
            scores = scores / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        if self.capture_attention:
            self.last_attention = attn.detach().cpu()
        else:
            self.last_attention = None
        context = self.attn_drop(attn) @ values
        context = context.transpose(1, 2).contiguous().reshape(batch, t_query, self.num_heads * self.d_v)
        return self.proj_drop(self.o_proj(context))


class PositionwiseFFN(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, attn_dropout, activation,
                 d_k=None, d_v=None, attention_scale=True, norm_style="post"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout, d_k, d_v, attention_scale)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_style = norm_style

    def forward(self, x, src_mask):
        if self.norm_style == "pre":
            x = x + self.drop(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
            x = x + self.drop(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
            x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, attn_dropout, activation,
                 d_k=None, d_v=None, attention_scale=True, norm_style="post"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attn_dropout, d_k, d_v, attention_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, attn_dropout, d_k, d_v, attention_scale)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_style = norm_style

    def forward(self, x, mem, tgt_mask, mem_mask):
        if self.norm_style == "pre":
            n = self.norm1(x)
            x = x + self.drop(self.self_attn(n, n, n, tgt_mask))
            x = x + self.drop(self.cross_attn(self.norm2(x), mem, mem, mem_mask))
            x = x + self.drop(self.ffn(self.norm3(x)))
        else:
            x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
            x = self.norm2(x + self.drop(self.cross_attn(x, mem, mem, mem_mask)))
            x = self.norm3(x + self.drop(self.ffn(x)))
        return x


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
        max_len: int = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        attention_scale: bool = True,
        norm_style: str = "post",
        final_norm: bool = True,
    ):
        super().__init__()
        if norm_style not in ("pre", "post"):
            raise ValueError("norm_style must be pre or post")
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = (
            self.src_embed if share_embeddings else nn.Embedding(vocab_size, d_model)
        )
        self.pos_enc_src = PositionalEncoding(d_model, max_len, dropout, pos_encoding)
        self.pos_enc_tgt = PositionalEncoding(d_model, max_len, dropout, pos_encoding)

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    d_model, num_heads, d_ff, dropout, attn_dropout, activation,
                    d_k, d_v, attention_scale, norm_style
                )
                for _ in range(N)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    d_model, num_heads, d_ff, dropout, attn_dropout, activation,
                    d_k, d_v, attention_scale, norm_style
                )
                for _ in range(N)
            ]
        )
        self.norm_enc = nn.LayerNorm(d_model) if final_norm else nn.Identity()
        self.norm_dec = nn.LayerNorm(d_model) if final_norm else nn.Identity()

        self.generator = nn.Linear(d_model, vocab_size)
        if (
            tie_softmax_weight
            and self.tgt_embed.weight.shape == self.generator.weight.shape
        ):
            self.generator.weight = self.tgt_embed.weight  # weight tying

        self.d_model = d_model
        self.vocab_size = vocab_size

    def encode(
        self, src_ids: torch.Tensor, src_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
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
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.pos_enc_tgt(self.tgt_embed(tgt_ids) * math.sqrt(self.d_model))
        # subsequent mask ∧ padding mask
        T = tgt_ids.size(1)
        sub_mask = torch.triu(
            torch.ones((1, 1, T, T), device=tgt_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        tgt_mask = (~sub_mask) & tgt_key_padding_mask.unsqueeze(1).unsqueeze(
            2
        )  # True=allowed
        mem_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.decoder:
            y = layer(y, mem, tgt_mask, mem_mask)
        return self.norm_dec(y)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        mem = self.encode(src_ids, src_key_padding_mask)
        out = self.decode(tgt_in_ids, mem, tgt_key_padding_mask, src_key_padding_mask)
        logits = self.generator(out)
        return logits
