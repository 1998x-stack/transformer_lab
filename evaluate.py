# transformer_lab/evaluate.py
from __future__ import annotations
import torch
from typing import List
from datasets import DatasetDict
from utils.config import load_config
from utils.logging_utils import setup_logging
from utils.metrics import compute_bleu
from utils.distributed import set_seed
from utils.torch_utils import create_padding_mask
from models.transformer import Transformer
from data.datasets import load_mt_dataset, filter_and_rename
from data.tokenization import train_or_load_spm
from utils.decoding import beam_search
from loguru import logger

@torch.no_grad()
def main():
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    args = p.parse_args()

    cfg = load_config(args.config)
    tb = setup_logging(cfg.runtime.tb_dir + "_eval")
    set_seed(cfg.runtime.seed)

    src_lang, tgt_lang = cfg.data.lang_pair.split("-")
    ddict: DatasetDict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
    ddict = filter_and_rename(ddict, src_lang, tgt_lang, cfg.data.min_len, cfg.data.max_src_len, cfg.data.max_tgt_len)
    tok = train_or_load_spm(ddict["train"], cfg.data.tokenizer_dir, cfg.data.vocab_size)

    model = Transformer(
        vocab_size=tok.vocab_size,
        N=cfg.model.N, d_model=cfg.model.d_model, d_ff=cfg.model.d_ff, num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout, attn_dropout=cfg.model.attn_dropout, activation=cfg.model.activation,
        share_embeddings=cfg.model.share_embeddings, tie_softmax_weight=cfg.model.tie_softmax_weight,
        pos_encoding=cfg.model.pos_encoding
    ).to(cfg.runtime.device)
    state = torch.load(args.ckpt, map_location=cfg.runtime.device)
    model.load_state_dict(state["model"])
    model.eval()

    refs, preds = [], []
    split = ddict[args.split]
    for ex in split:
        src = ex["src"]
        ref = ex["tgt"]
        src_ids = torch.tensor([tok.encode(src)], device=cfg.runtime.device)
        hyp_ids = beam_search(
            model, src_ids, tok.pad_id, tok.bos_id, tok.eos_id,
            beam=cfg.decode.beam_size, alpha=cfg.decode.length_penalty,
            max_len_ratio=cfg.decode.max_len_ratio, max_len_offset=cfg.decode.max_len_offset
        )[0]
        hyp = tok.decode(hyp_ids)
        refs.append(ref)
        preds.append(hyp)
        if len(preds) % 200 == 0:
            logger.info(f"decoded {len(preds)} samples...")

    bleu = compute_bleu(preds, refs)
    logger.info(f"{args.split} sacreBLEU = {bleu:.2f}")

if __name__ == "__main__":
    main()
