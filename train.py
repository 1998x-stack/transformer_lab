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
from utils.torch_utils import create_padding_mask, count_parameters, resolve_device
from utils.metrics import SpeedMeter, perplexity
from optim.scheduler import NoamScheduler
from models.factory import build_model, build_training_components
from research.tracking import RunRecorder
import torch.nn.functional as F
from models.label_smoothing import LabelSmoothingLoss
from data.datasets import load_mt_dataset, filter_and_rename, load_copy_corpus
from data.tokenization import train_or_load_spm, SPTokenizer
from data.collate import DynamicBatcher, collate

def train_loop(cfg: TrainConfig):
    tb = setup_logging(cfg.runtime.tb_dir)
    set_seed(cfg.runtime.seed)

    # 语言对解析
    src_lang, tgt_lang = cfg.data.lang_pair.split("-")
    # 数据：支持本地 copy 语料 (src==tgt) 与 HF 平行语料两种模式
    if cfg.data.dataset == "copy":
        ddict: DatasetDict = load_copy_corpus(cfg.data.corpus_path, cfg.data.min_len, cfg.data.max_src_len, seed=cfg.runtime.seed)
    else:
        # 加载数据
        ddict: DatasetDict = load_mt_dataset(cfg.data.dataset, cfg.data.lang_pair, cfg.data.cache_dir)
        ddict = filter_and_rename(ddict, src_lang, tgt_lang, cfg.data.min_len, cfg.data.max_src_len, cfg.data.max_tgt_len)

    tok: SPTokenizer = train_or_load_spm(ddict["train"], cfg.data.tokenizer_dir, cfg.data.vocab_size)
    pad_id, bos_id, eos_id = tok.pad_id, tok.bos_id, tok.eos_id

    device = resolve_device(cfg.runtime.device)
    use_amp = cfg.runtime.amp and device.type == "cuda"
    logger.info(f"Training on device={device} amp={use_amp}")

    # 模型
    model = build_model(cfg.model, tok.vocab_size).to(device)

    logger.info(f"Model params: {count_parameters(model):,}")

    opt, sched, criterion = build_training_components(cfg, model, tok.vocab_size, pad_id)
    recorder = RunRecorder(cfg, count_parameters(model)) if is_main_process() else None

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    speed = SpeedMeter()

    # 动态批处理器
    train_iter = DynamicBatcher(ddict["train"], tok, cfg.data.max_tokens_per_batch, cfg.data.num_buckets, shuffle=True)
    global_step = 0
    best_loss = float("inf")
    micro_steps = 0
    accum_target_tokens = 0
    log_target_tokens = 0
    log_smoothed_sum = 0.0
    log_nll_sum = 0.0
    if cfg.runtime.accumulate_steps < 1:
        raise ValueError("accumulate_steps must be at least one")
    opt.zero_grad(set_to_none=True)
    last_ckpts = []

    Path(cfg.runtime.ckpt_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(10**9):  # 直到达到 max_steps
        for samples in train_iter:
            batch = collate(samples, pad_id)
            src_ids = batch["src_ids"].to(device)
            tgt_in_ids = batch["tgt_in_ids"].to(device)
            tgt_out_ids = batch["tgt_out_ids"].to(device)

            src_mask = create_padding_mask(src_ids, pad_id)
            tgt_mask = create_padding_mask(tgt_in_ids, pad_id)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(src_ids, tgt_in_ids, src_mask, tgt_mask)  # (B,T,V)
                loss = criterion(logits, tgt_out_ids)

            # Sum token losses across heterogeneous microbatches before
            # reducing gradients by the actual non-padding target count.
            valid = int(tgt_out_ids.ne(pad_id).sum().item())
            if valid == 0:
                continue
            scaler.scale(loss * valid).backward()
            with torch.no_grad():
                nll_sum = F.cross_entropy(logits.detach().reshape(-1, logits.size(-1)),
                                          tgt_out_ids.reshape(-1), ignore_index=pad_id,
                                          reduction="sum").item()
            log_smoothed_sum += loss.item() * valid
            log_nll_sum += nll_sum
            log_target_tokens += valid
            accum_target_tokens += valid
            tokens = int((src_ids.ne(pad_id).sum() + tgt_in_ids.ne(pad_id).sum()).item())
            speed.update(tokens)
            micro_steps += 1
            if micro_steps % cfg.runtime.accumulate_steps:
                continue
            scaler.unscale_(opt)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(accum_target_tokens)
            if cfg.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            effective_target_tokens = accum_target_tokens
            accum_target_tokens = 0
            global_step += 1

            if is_main_process() and global_step % cfg.runtime.log_every == 0:
                lr = sched.get_last_lr()[0]
                mean_loss = log_smoothed_sum / log_target_tokens
                nll = log_nll_sum / log_target_tokens
                ppl = math.exp(min(80, nll))
                tb.add_scalar("train/loss", mean_loss, global_step)
                tb.add_scalar("train/ppl", ppl, global_step)
                tb.add_scalar("train/nll", nll, global_step)
                tb.add_scalar("train/lr", lr, global_step)
                tb.add_scalar("train/tokens_per_sec", speed.rate(), global_step)
                tb.add_scalar("train/target_tokens", log_target_tokens, global_step)
                recorder.record("train", global_step, loss=mean_loss, nll=nll,
                                ppl=ppl, lr=lr, tokens_per_sec=speed.rate(),
                                target_tokens=log_target_tokens,
                                target_tokens_last_update=effective_target_tokens)
                logger.info(f"step={global_step} loss={mean_loss:.4f} nll={nll:.4f} ppl={ppl:.2f} lr={lr:.6f} tok/s={speed.rate():.0f}")
                log_smoothed_sum = log_nll_sum = 0.0
                log_target_tokens = 0

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
                    n_batch, tot_loss, tot_nll, tot_tokens = 0, 0.0, 0.0, 0
                    for samples in dev_iter:
                        batch = collate(samples[:16], pad_id)  # 小批
                        src_ids = batch["src_ids"].to(device)
                        tgt_in_ids = batch["tgt_in_ids"].to(device)
                        tgt_out_ids = batch["tgt_out_ids"].to(device)
                        src_mask = src_ids.ne(pad_id)
                        tgt_mask = tgt_in_ids.ne(pad_id)
                        logits = model(src_ids, tgt_in_ids, src_mask, tgt_mask)
                        l = criterion(logits, tgt_out_ids).item()
                        valid = int(tgt_out_ids.ne(pad_id).sum().item())
                        tot_loss += l * valid
                        tot_nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       tgt_out_ids.reshape(-1), ignore_index=pad_id,
                                       reduction="sum").item()
                        tot_tokens += valid
                        n_batch += 1
                        if n_batch >= 10:
                            break
                    val_loss = tot_loss / max(1, tot_tokens)
                    val_nll = tot_nll / max(1, tot_tokens)
                    val_ppl = math.exp(min(80, val_nll))
                model.train()
                if is_main_process():
                    tb.add_scalar("val/loss", val_loss, global_step)
                    tb.add_scalar("val/nll", val_nll, global_step)
                    tb.add_scalar("val/ppl", val_ppl, global_step)
                    recorder.record("validation", global_step, loss=val_loss,
                                    nll=val_nll, ppl=val_ppl, evaluated_tokens=tot_tokens)
                    logger.info(f"[eval] step={global_step} val_loss={val_loss:.4f} val_nll={val_nll:.4f} val_ppl={val_ppl:.2f}")
                    if val_nll < best_loss:
                        best_loss = val_nll
                        torch.save({"model": model.state_dict(), "step": global_step}, f"{cfg.runtime.ckpt_dir}/best.pt")

            if global_step >= cfg.optim.max_steps:
                logger.info("Reached max_steps. Stop.")
                tb.flush()
                tb.close()
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
