"""Stage 0 — Continued MLM pretraining on AMP positives + UniProt short peptides.

8x V100 DDP via accelerate. ESM-2 650M backbone wrapped as AutoModelForMaskedLM +
LoRA on q/k/v/dense, gradient checkpointing enabled, fp16.

Launch:
  accelerate launch --num_processes 8 --num_machines 1 --mixed_precision fp16 \\
      -m amp_prioritizer.mlm_pretrain --config configs/stage0_mlm.yaml

Memory budget (V100 32GB, batch_size=8 per device, max_len=200):
  - ESM-2 650M fp16 weights: ~1.3 GB
  - LoRA params (r=16, q/k/v/dense): ~16 MB
  - activations (grad-ckpt enabled): ~10 GB at batch 8
  - AdamW state (LoRA params): ~64 MB
  - total ≈ 12 GB / GPU, 8 GPUs → global batch = 8 * 8 * grad_accum=2 = 128
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from amp_prioritizer.schema import sanitize_seq


class MLMDataset(Dataset):
    def __init__(self, seqs: list[str], max_len: int = 200):
        self.seqs = [s for s in (sanitize_seq(x) for x in seqs) if s]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict:
        return {"seq": self.seqs[idx][: self.max_len]}


def collator_with_masking(tokenizer, max_len: int, mask_ratio: float = 0.15,
                          mask_span: int = 3):
    """Random span-masking collator. seed-aware (RNG controlled by accelerate.set_seed)."""
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    def coll(batch):
        seqs = [b["seq"] for b in batch]
        enc = tokenizer(seqs, padding=True, truncation=True,
                        max_length=max_len + 2, return_tensors="pt")
        input_ids = enc["input_ids"].clone()
        attn = enc["attention_mask"]
        labels = input_ids.clone()
        labels.fill_(-100)
        for i in range(input_ids.shape[0]):
            valid = (attn[i] == 1).nonzero(as_tuple=True)[0]
            # skip BOS / EOS
            if len(valid) < 2:
                continue
            valid = valid[(valid != 0) & (valid != valid.max())]
            if len(valid) == 0:
                continue
            n_mask = max(1, int(mask_ratio * len(valid)))
            chosen = valid[torch.randperm(len(valid))[:n_mask]]
            for pos in chosen.tolist():
                span = int(np.random.randint(1, mask_span + 1))
                end = min(pos + span, input_ids.shape[1] - 1)
                labels[i, pos:end] = input_ids[i, pos:end]
                input_ids[i, pos:end] = mask_id
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return coll


def _read_fasta(path: str | Path) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def load_seqs(cfg: dict, verbose: bool = True) -> tuple[list[str], list[str]]:
    """Build the MLM corpus = AMP+ from peptides.parquet ⊕ a single CD-HIT'd
    PeptideAtlas FASTA (`data.mlm_corpus_fasta`). Missing FASTA → AMP-only."""
    df = pd.read_parquet(cfg["data"]["parquet"])
    amp_pos = df["seq"].drop_duplicates().tolist()
    if verbose:
        print(f"AMP positives from parquet: {len(amp_pos)}")

    corpus_path = cfg["data"].get("mlm_corpus_fasta")
    extra_seqs: list[str] = []
    if corpus_path and Path(corpus_path).exists():
        extra_seqs = _read_fasta(corpus_path)
        if verbose:
            print(f"loaded {corpus_path}: +{len(extra_seqs)} seqs")
    elif verbose:
        print(f"mlm_corpus_fasta missing ({corpus_path}); training on AMP-only")

    # Cross-source dedup via set() + sanitize in one pass (enforces 3–200 aa
    # + canonical alphabet via amp_prioritizer.schema::sanitize_seq).
    all_seqs = sorted({s for s in (sanitize_seq(x) for x in amp_pos + extra_seqs) if s})
    if verbose:
        print(f"total after sanitize+dedup: {len(all_seqs)}")

    rng = np.random.default_rng(cfg.get("seed", 42))
    perm = rng.permutation(len(all_seqs))
    cut = int(len(all_seqs) * 0.95)
    train = [all_seqs[i] for i in perm[:cut]]
    val = [all_seqs[i] for i in perm[cut:]]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="override per-device batch")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "fp16"),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        log_with=cfg.get("log_with"),
        # LoRA freezes the base model → DDP does not put frozen parameters into a bucket, so
        # find_unused_parameters is not needed; enabling it would actually conflict with the
        # reentrant backward of gradient checkpointing.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    set_seed(cfg.get("seed", 42))

    out_dir = Path(cfg.get("out_dir", "runs/stage0_mlm"))
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"== Stage 0 MLM config ==")
        print(f"world_size={accelerator.num_processes}  "
              f"mixed_precision={accelerator.mixed_precision}  "
              f"grad_accum={accelerator.gradient_accumulation_steps}")
        print(f"out_dir={out_dir}")

    # ---------- model ----------
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"],
                                        cache_dir=cfg.get("hf_cache_dir"))
    model = AutoModelForMaskedLM.from_pretrained(cfg["model"]["name"],
                                                  cache_dir=cfg.get("hf_cache_dir"))
    # Enable gradient checkpointing — mandatory for DDP + LoRA + 650M MLM, otherwise it will not
    # fit in V100 32GB.
    # use_reentrant=False is required under DDP: the reentrant version reruns the forward pass during
    # backward, causing DDP to trigger grad reduce twice for the same parameter and raise
    # "marked a variable ready only once".
    if cfg["model"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=cfg["model"]["lora_r"], lora_alpha=cfg["model"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora_dropout"], bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=["query", "key", "value", "dense"],
    )
    model = get_peft_model(model, lora_cfg)
    # PEFT wraps the model in another layer, so grad-ckpt must be re-enabled (again forcing use_reentrant=False)
    if cfg["model"].get("gradient_checkpointing", True):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  # grad-ckpt requires input emb grads on

    if accelerator.is_main_process:
        tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"trainable / total: {tr/1e6:.1f}M / {tot/1e6:.1f}M ({tr/tot:.1%})")

    # ---------- data ----------
    train_seqs, val_seqs = load_seqs(cfg, verbose=accelerator.is_main_process)
    train_ds = MLMDataset(train_seqs, max_len=cfg["data"]["max_len"])
    val_ds = MLMDataset(val_seqs, max_len=cfg["data"]["max_len"])
    coll = collator_with_masking(tok, max_len=cfg["data"]["max_len"],
                                  mask_ratio=cfg["train"].get("mask_ratio", 0.15),
                                  mask_span=cfg["train"].get("mask_span", 3))
    bsz = args.batch_size or cfg["train"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=coll, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bsz, shuffle=False,
        num_workers=2, collate_fn=coll, pin_memory=True,
    )
    if accelerator.is_main_process:
        global_bsz = bsz * accelerator.num_processes * accelerator.gradient_accumulation_steps
        print(f"per-device batch={bsz}, world={accelerator.num_processes}, "
              f"grad_accum={accelerator.gradient_accumulation_steps} → global batch={global_bsz}")
        print(f"train n={len(train_ds)}  val n={len(val_ds)}")

    # ---------- optim + sched ----------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("wd", 0.01))
    total_steps = args.max_steps or (
        cfg["train"]["epochs"] * len(train_loader)
        // (accelerator.gradient_accumulation_steps * accelerator.num_processes)
    )
    warmup = int(total_steps * cfg["train"].get("warmup_ratio", 0.05))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model, opt, train_loader, val_loader, sched = accelerator.prepare(
        model, opt, train_loader, val_loader, sched
    )

    # ---------- loop ----------
    best_pppl = float("inf")
    step = 0
    log_every = cfg["train"].get("log_every", 50)
    eval_every = cfg["train"].get("eval_every", 2000)
    save_every = cfg["train"].get("save_every", eval_every)
    t0 = time.time()

    while step < total_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                loss = out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                cfg["train"].get("grad_clip", 1.0))
                opt.step()
                sched.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                # accelerator.gather averages the loss across all ranks
                if step % log_every == 0:
                    loss_avg = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
                    if accelerator.is_main_process:
                        dt = time.time() - t0
                        print(f"[mlm {step:6d}/{total_steps}] loss={loss_avg:.3f} "
                              f"ppl={math.exp(min(loss_avg, 20)):.2f} "
                              f"lr={sched.get_last_lr()[0]:.2e} {step/dt:.1f}step/s")

                if step % eval_every == 0 or step >= total_steps:
                    model.eval()
                    val_losses: list[float] = []
                    with torch.no_grad():
                        for vb in val_loader:
                            v_out = model(**vb)
                            val_losses.append(v_out.loss.detach())
                    # gather across ranks
                    if val_losses:
                        v_stack = torch.stack(val_losses)
                        v_gathered = accelerator.gather(v_stack)
                        pppl = math.exp(min(v_gathered.mean().item(), 20))
                    else:
                        pppl = float("inf")

                    if accelerator.is_main_process:
                        print(f"  [eval @ {step}] val pseudo-ppl = {pppl:.3f}")
                        if pppl < best_pppl:
                            best_pppl = pppl
                            unwrapped = accelerator.unwrap_model(model)
                            torch.save({
                                "model": unwrapped.state_dict(),
                                "step": step, "pppl": pppl,
                            }, out_dir / "best.pt")
                            print(f"  saved best (pppl={pppl:.3f}) -> {out_dir/'best.pt'}")
                    accelerator.wait_for_everyone()
                    model.train()

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save({"model": unwrapped.state_dict()}, out_dir / "last.pt")
        print(f"done. best pppl = {best_pppl:.3f}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
