# transformer_lab/data/tokenization.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import sentencepiece as spm
from datasets import Dataset
import os, itertools

class SPTokenizer:
    """Shared vocabulary SentencePiece tokenizer."""
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        if self.pad_id < 0:
            # 若用户训练时未设置 pad/bos/eos，回退：添加特殊符号映射
            self.pad_id = 0

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

def train_or_load_spm(train_ds: Dataset, save_dir: str, vocab_size: int = 37000) -> SPTokenizer:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(save_dir) / f"spm_{vocab_size}.model"
    if model_path.exists():
        return SPTokenizer(str(model_path))
    # 将 src+tgt 拼接作为训练语料
    txt_path = Path(save_dir) / "all.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for rec in train_ds:
            f.write(rec["src"] + "\n")
            f.write(rec["tgt"] + "\n")
    spm.SentencePieceTrainer.train(
        input=str(txt_path),
        model_prefix=str(model_path).replace(".model", ""),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        user_defined_symbols=[]
    )
    return SPTokenizer(str(model_path))
