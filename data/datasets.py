# transformer_lab/data/datasets.py
from __future__ import annotations
from typing import Dict, Tuple, Iterable, List
from datasets import load_dataset, load_from_disk, DatasetDict
import os

def load_mt_dataset(dataset: str, lang_pair: str, cache_dir: str | None = None) -> DatasetDict:
    """Load translation dataset; try local path then web."""
    subset = lang_pair.replace("-", "")
    # 尝试本地 HF 缓存目录（若用户提前下载）
    local_hint = os.environ.get("HF_LOCAL_DATASET_DIR")
    if local_hint and os.path.isdir(local_hint):
        try:
            return load_from_disk(local_hint)
        except Exception:
            pass
    # 公开数据集名称映射
    name = dataset
    if dataset == "wmt14":
        # wmt14 en-de 在HF中名为 'wmt14' 配置 'de-en' 或 'en-fr' 等
        pass
    ddict = load_dataset(name, lang_pair, cache_dir=cache_dir)
    return ddict

def filter_and_rename(ddict: DatasetDict, src_lang: str, tgt_lang: str, min_len: int, max_src: int, max_tgt: int) -> DatasetDict:
    def _proc(ex):
        s = ex["translation"][src_lang].strip()
        t = ex["translation"][tgt_lang].strip()
        return {"src": s, "tgt": t, "src_len": len(s.split()), "tgt_len": len(t.split())}
    ddict = ddict.map(_proc, remove_columns=ddict["train"].column_names, num_proc=1)
    ddict = ddict.filter(lambda ex: (min_len <= ex["src_len"] <= max_src) and (min_len <= ex["tgt_len"] <= max_tgt))
    return ddict
