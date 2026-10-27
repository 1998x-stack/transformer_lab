# transformer_lab/decode.py
from __future__ import annotations
from pathlib import Path
import torch
from utils.config import load_config
from utils.torch_utils import resolve_device
from data.tokenization import train_or_load_spm
from data.datasets import load_mt_dataset, load_copy_corpus
from models.factory import build_model
from utils.decoding import beam_search

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--src", type=str, required=True, help="source sentence to translate")
    p.add_argument("--attention-dir", type=str, default=None,
                   help="directory to save attention heatmaps from decoding")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.runtime.device)
    ddict = (load_copy_corpus(cfg.data.corpus_path, cfg.data.min_len, cfg.data.max_src_len, seed=cfg.runtime.seed)
             if cfg.data.dataset == "copy"
             else load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir))
    tok = train_or_load_spm(ddict["train"], cfg.data.tokenizer_dir, cfg.data.vocab_size)

    model = build_model(cfg.model, tok.vocab_size).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    if args.attention_dir:
        from research.visualize import capture_attention
        capture_attention(model)

    src_ids = torch.tensor([tok.encode(args.src)], device=device)
    out_ids = beam_search(model, src_ids, tok.pad_id, tok.bos_id, tok.eos_id,
                          beam=cfg.decode.beam_size, alpha=cfg.decode.length_penalty,
                          max_len_ratio=cfg.decode.max_len_ratio, max_len_offset=cfg.decode.max_len_offset,
                          repeat_penalty=cfg.decode.repeat_penalty)[0]
    print(tok.decode(out_ids))
    if args.attention_dir:
        from research.visualize import attention_heatmaps
        for path in attention_heatmaps(model, Path(args.attention_dir)):
            print(f"attention heatmap: {path}")

if __name__ == "__main__":
    main()
