# transformer_lab/decode.py
from __future__ import annotations
import torch
from utils.config import load_config
from data.tokenization import train_or_load_spm
from data.datasets import load_mt_dataset, filter_and_rename
from models.transformer import Transformer
from utils.decoding import beam_search

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--src", type=str, required=True, help="source sentence to translate")
    args = p.parse_args()

    cfg = load_config(args.config)
    src_lang, tgt_lang = cfg.data.lang_pair.split("-")
    ddict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
    tok = train_or_load_spm(ddict["train"], cfg.data.tokenizer_dir, cfg.data.vocab_size)

    model = Transformer(vocab_size=tok.vocab_size, N=cfg.model.N, d_model=cfg.model.d_model, d_ff=cfg.model.d_ff,
                        num_heads=cfg.model.num_heads, dropout=cfg.model.dropout, attn_dropout=cfg.model.attn_dropout,
                        activation=cfg.model.activation, share_embeddings=cfg.model.share_embeddings,
                        tie_softmax_weight=cfg.model.tie_softmax_weight, pos_encoding=cfg.model.pos_encoding
                        ).to(cfg.runtime.device)
    state = torch.load(args.ckpt, map_location=cfg.runtime.device)
    model.load_state_dict(state["model"])
    model.eval()

    src_ids = torch.tensor([tok.encode(args.src)], device=cfg.runtime.device)
    out_ids = beam_search(model, src_ids, tok.pad_id, tok.bos_id, tok.eos_id,
                          beam=cfg.decode.beam_size, alpha=cfg.decode.length_penalty,
                          max_len_ratio=cfg.decode.max_len_ratio, max_len_offset=cfg.decode.max_len_offset)[0]
    print(tok.decode(out_ids))

if __name__ == "__main__":
    main()
