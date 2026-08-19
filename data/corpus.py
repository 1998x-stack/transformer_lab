# transformer_lab/data/corpus.py
from __future__ import annotations
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict


def parse_copy_corpus(path: str, min_len: int, max_len: int) -> List[Dict[str, str]]:
    """Read a plain-text file into ``src == tgt`` copy records.

    Skips blank lines, ALL-CAPS headings/dividers, and lines outside
    ``[min_len, max_len]`` words so the demo corpus is clean sentence-like prose.
    """
    records: List[Dict[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        text = " ".join(raw.split())
        n_words = len(text.split())
        if not text or n_words < min_len or n_words > max_len:
            continue
        if all(c.isupper() or c in " .,;:!?\"'()[]-“”‘’" for c in text):
            continue
        records.append({"src": text, "tgt": text})
    return records


def load_copy_corpus(
    corpus_path: str,
    min_len: int,
    max_len: int,
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
) -> DatasetDict:
    """Build a deterministic train/validation/test split of copy records."""
    records = parse_copy_corpus(corpus_path, min_len, max_len)
    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = records[:n_test]
    val = records[n_test : n_test + n_val]
    train = records[n_test + n_val :]
    return DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
            "test": Dataset.from_list(test),
        }
    )