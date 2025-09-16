# transformer_lab/data/collate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch
import math
import random

@dataclass
class Sample:
    src_ids: List[int]
    tgt_ids: List[int]

def pad_sequences(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def collate(samples: List[Sample], pad_id: int) -> Dict[str, torch.Tensor]:
    src = [s.src_ids for s in samples]
    tgt = [s.tgt_ids for s in samples]
    # decoder 输入输出错一位
    tgt_in = [ [*t[:-1]] for t in tgt ]
    tgt_out = [ [*t[1:]] for t in tgt ]
    batch = {
        "src_ids": pad_sequences(src, pad_id),
        "tgt_in_ids": pad_sequences(tgt_in, pad_id),
        "tgt_out_ids": pad_sequences(tgt_out, pad_id),
    }
    return batch

class DynamicBatcher:
    """按近似长度分桶 & 动态 token 数配额.
    论文口径: ~25k 源 + 25k 目标 tokens / batch.
    """
    def __init__(self, dataset, tokenizer, max_tokens: int = 50_000, num_buckets: int = 8, shuffle: bool = True):
        self.ds = dataset
        self.tok = tokenizer
        self.max_tokens = max_tokens
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        # 提前编码长度用于分桶
        self.lengths = [max(len(self.tok.encode(r["src"])), len(self.tok.encode(r["tgt"]))) for r in self.ds]

    def __iter__(self):
        idxs = list(range(len(self.ds)))
        if self.shuffle:
            random.shuffle(idxs)
        # 分桶
        idxs.sort(key=lambda i: self.lengths[i])
        buckets = [idxs[i::self.num_buckets] for i in range(self.num_buckets)]
        for b in buckets:
            start = 0
            while start < len(b):
                batch_idx = []
                tokens = 0
                max_len_src = 1
                max_len_tgt = 1
                j = start
                while j < len(b):
                    ex = self.ds[b[j]]
                    src_ids = self.tok.encode(ex["src"])
                    tgt_ids = self.tok.encode(ex["tgt"], add_bos=True, add_eos=True)
                    max_len_src = max(max_len_src, len(src_ids))
                    max_len_tgt = max(max_len_tgt, len(tgt_ids))
                    prospective = (len(batch_idx) + 1) * (max_len_src + max_len_tgt)
                    if prospective > self.max_tokens and batch_idx:
                        break
                    batch_idx.append(b[j])
                    j += 1
                # 产出 batch
                samples = []
                for k in batch_idx:
                    ex = self.ds[k]
                    samples.append(Sample(self.tok.encode(ex["src"]),
                                          self.tok.encode(ex["tgt"], add_bos=True, add_eos=True)))
                yield samples
                start = j
