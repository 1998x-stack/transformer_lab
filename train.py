# transformer_lab/train.py
from __future__ import annotations
import os, math, time
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.optim import Adam
from loguru import logger
from datasets import DatasetDict
from tqdm import tqdm

from utils.config import load_config, TrainConfig
from utils.logging_utils import setup_logging
from utils.distributed import set_seed, is_main_process
from utils.torch_utils import create_padding_mask, count_parameters
from utils.metrics import SpeedMeter, perplexity
from optim.scheduler import NoamScheduler
from models.transformer import Transformer
from models.label_smoothing import LabelSmoothingLoss
from data.datasets import load_mt_dataset, filter_and_rename
from data.tokenization import train_or_load_spm, SPTokenizer
from data.collate import DynamicBatcher, collate

def train_loop(cfg: TrainConfig):
    tb = setup_logging(cfg.runtime.tb_dir)
    set_seed(cfg.runtime.seed)

    # 语言对解析
    src_lang, tgt_lang = cfg.data.lang_pair.split("-")
    # 加载数据
    ddict: DatasetDict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
    ddict = filter_and_rename(ddict, src_lang, tgt_lang, cfg.data.min_len, cfg.data.max_src_len, cfg.data.max_tgt_len)

    tok: SPTokenizer = train_or_load_spm(ddict["train"], cfg.data.tokenizer_dir, cfg.data.vocab_size)
    pad_id, bos_id, eos_id = tok.pad_id, tok.bos_id, tok.eos_id

    # 模型
    model = Transformer(
        vocab_size=tok.vocab_size,
        N=cfg.model.N,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        attn_dropout=cfg.model.attn_dropout,
        activation=cfg.model.activation,
        share_embeddings=cfg.model.share_embeddings,
        tie_softmax_weight=cfg.model.tie_softmax_weight,
        pos_encoding=cfg.model.pos_encoding
    ).to(cfg.runtime.device)

    logger.info(f"Model params: {count_parameters(model):,}")

    opt = Adam(model.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas, eps=cfg.optim.eps, weight_decay=cfg.optim.weight_decay)
    sched = NoamScheduler(opt, d_model=cfg.model.d_model, warmup_steps=cfg.optim.warmup_steps)
    criterion = LabelSmoothingLoss(classes=tok.vocab_size, smoothing=cfg.model.label_smoothing, ignore_index=pad_id)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.runtime.amp)
    speed = SpeedMeter()

    # 动态批处理器
    train_iter = DynamicBatcher(ddict["train"], tok, cfg.data.max_tokens_per_batch, cfg.data.num_buckets, shuffle=True)
    global_step = 0
    best_loss = float("inf")
    last_ckpts = []

    Path(cfg.runtime.ckpt_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(10**9):  # 直到达到 max_steps
        for samples in train_iter:
            batch = collate(samples, pad_id)
            src_ids = batch["src_ids"].to(cfg.runtime.device)
            tgt_in_ids = batch["tgt_in_ids"].to(cfg.runtime.device)
            tgt_out_ids = batch["tgt_out_ids"].to(cfg.runtime.device)

            src_mask = create_padding_mask(src_ids, pad_id)
            tgt_mask = create_padding_mask(tgt_in_ids, pad_id)

            with torch.cuda.amp.autocast(enabled=cfg.runtime.amp):
                logits = model(src_ids, tgt_in_ids, src_mask, tgt_mask)  # (B,T,V)
                loss = criterion(logits, tgt_out_ids)

            scaler.scale(loss).backward()
            if cfg.optim.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()

            # 统计
            tokens = int((src_ids.ne(pad_id).sum() + tgt_in_ids.ne(pad_id).sum()).item())
            speed.update(tokens)
            global_step += 1

            if is_main_process() and global_step % 100 == 0:
                lr = sched.get_last_lr()[0]
                ppl = perplexity(loss.item())
                tb.add_scalar("train/loss", loss.item(), global_step)
                tb.add_scalar("train/ppl", ppl, global_step)
                tb.add_scalar("train/lr", lr, global_step)
                tb.add_scalar("train/tokens_per_sec", speed.rate(), global_step)
                logger.info(f"step={global_step} loss={loss.item():.4f} ppl={ppl:.2f} lr={lr:.6f} tok/s={speed.rate():.0f}")

            # 保存与评估
            if is_main_process() and global_step % cfg.runtime.save_every == 0:
                ckpt_path = f"{cfg.runtime.ckpt_dir}/step{global_step}.pt"
                torch.save({"model": model.state_dict(), "step": global_step}, ckpt_path)
                last_ckpts.append(ckpt_path)
                if len(last_ckpts) > cfg.runtime.keep_last:
                    os.remove(last_ckpts.pop(0))

            if global_step % cfg.runtime.eval_every == 0:
                # 简要验证：用 teacher-forcing loss 评估一小批 dev
                model.eval()
                with torch.no_grad():
                    dev_iter = DynamicBatcher(ddict["validation"], tok, cfg.data.max_tokens_per_batch // 2, cfg.data.num_buckets, shuffle=False)
                    n_batch, tot_loss = 0, 0.0
                    for samples in dev_iter:
                        batch = collate(samples[:16], pad_id)  # 小批
                        src_ids = batch["src_ids"].to(cfg.runtime.device)
                        tgt_in_ids = batch["tgt_in_ids"].to(cfg.runtime.device)
                        tgt_out_ids = batch["tgt_out_ids"].to(cfg.runtime.device)
                        src_mask = src_ids.ne(pad_id)
                        tgt_mask = tgt_in_ids.ne(pad_id)
                        logits = model(src_ids, tgt_in_ids, src_mask, tgt_mask)
                        l = criterion(logits, tgt_out_ids).item()
                        tot_loss += l
                        n_batch += 1
                        if n_batch >= 10:
                            break
                    val_loss = tot_loss / max(1, n_batch)
                model.train()
                if is_main_process():
                    tb.add_scalar("val/loss", val_loss, global_step)
                    tb.add_scalar("val/ppl", math.exp(val_loss), global_step)
                    logger.info(f"[eval] step={global_step} val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}")
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save({"model": model.state_dict(), "step": global_step}, f"{cfg.runtime.ckpt_dir}/best.pt")

            if global_step >= cfg.optim.max_steps:
                logger.info("Reached max_steps. Stop.")
                return

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    train_loop(cfg)

if __name__ == "__main__":
    main()