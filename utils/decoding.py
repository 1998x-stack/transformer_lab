# transformer_lab/utils/decoding.py
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple
import torch
from ..models.transformer import Transformer
from .torch_utils import create_padding_mask

@dataclass
class BeamHypo:
    tokens: List[int]
    logprob: float
    ended: bool

def length_penalty(len_y: int, alpha: float) -> float:
    return ((5 + len_y) / 6) ** alpha

@torch.no_grad()
def beam_search(model: Transformer, src_ids: torch.Tensor, src_pad_id: int,
                bos_id: int, eos_id: int, beam: int = 4, alpha: float = 0.6,
                max_len_ratio: float = 1.0, max_len_offset: int = 50) -> List[List[int]]:
    device = src_ids.device
    B, S = src_ids.shape
    src_mask = create_padding_mask(src_ids, src_pad_id)
    mem = model.encode(src_ids, src_mask)

    max_len = int(S * max_len_ratio + max_len_offset)
    results: List[List[int]] = []

    for b in range(B):
        hypos = [BeamHypo(tokens=[bos_id], logprob=0.0, ended=False)]
        finished: List[BeamHypo] = []

        src_mask_b = src_mask[b:b+1]
        mem_b = mem[b:b+1]

        for t in range(1, max_len + 1):
            # 扩展所有 beam
            all_candidates: List[BeamHypo] = []
            for h in hypos:
                if h.ended:
                    all_candidates.append(h)
                    continue
                tgt = torch.tensor(h.tokens, device=device).unsqueeze(0)
                tgt_mask = create_padding_mask(tgt, src_pad_id)  # 这里 pad_id 不会被用到，因为无 pad
                dec = model.decode(tgt, mem_b, tgt_mask, src_mask_b)
                logits = model.generator(dec[:, -1, :])  # 最后一步
                logprobs = torch.log_softmax(logits, dim=-1).squeeze(0)
                topk_logprob, topk_idx = torch.topk(logprobs, beam)
                for lp, idx in zip(topk_logprob.tolist(), topk_idx.tolist()):
                    new_tokens = h.tokens + [idx]
                    ended = (idx == eos_id)
                    # 使用长度惩罚的归一化 logP
                    norm = (h.logprob + lp) / length_penalty(len(new_tokens), alpha)
                    all_candidates.append(BeamHypo(new_tokens, norm, ended))
            # 取前 beam
            all_candidates.sort(key=lambda x: x.logprob, reverse=True)
            hypos = all_candidates[:beam]
            # 移动已完成句子
            still_alive = []
            for h in hypos:
                if h.ended:
                    finished.append(h)
                else:
                    still_alive.append(h)
            hypos = still_alive or hypos
            # 早停：若达到 beam 个完成
            if len(finished) >= beam:
                break

        best = (finished or hypos)[0]
        # 去掉 BOS/EOS
        out = [tok for tok in best.tokens if tok not in (bos_id,)]
        if out and out[-1] == eos_id:
            out = out[:-1]
        results.append(out)
    return results
