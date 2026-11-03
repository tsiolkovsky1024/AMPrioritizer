"""Stage 3 — Latent CVAE + AR decoder + LengthHead training.

Input:  <cache>.npz + <cache>.species.json  (produced by scripts/cache_latents.py)
Output: runs/stage3_gen/best.pt

Loss = L_AR_CE + 0.1·L_length_CE + 0.5·L_recon_mse + β(step)·KL
  β cyclic-annealed via beta_schedule + free-bits floor 0.5 nat/dim.

Launch:
  python -m amp_prioritizer.gen.train_gen --config configs/stage3_gen.yaml \\
      --cache data/current/latent_cache.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


from amp_prioritizer.schema import load_latest_species
from amp_prioritizer.gen.cvae import (
    CVAEConfig, LENGTH_BUCKETS, beta_schedule,
    bucket_charge, bucket_length, kl_gaussian,
)
from amp_prioritizer.gen.generator import Generator
from amp_prioritizer.physchem import compute_one


class LatentCacheDataset(Dataset):
    """Reads (pooled, seq, species_mask, log10_mic) per row from the npz cache.
    Each __getitem__ picks one random active species as the conditioning signal
    (or a uniform-random species if the row has no MIC labels)."""

    def __init__(self, cache_path: str, tokenizer, n_species: int, max_len: int = 100):
        data = np.load(cache_path, allow_pickle=True)
        self.seqs = list(data["seq"])
        self.pooled = data["pooled"]
        self.species_mask = data["species_mask"]
        self.length = data["length"]
        self.tok = tokenizer
        self.max_len = max_len
        self.n_species = n_species
        if self.species_mask.shape[1] != n_species:
            raise ValueError(
                f"cache has {self.species_mask.shape[1]} species but "
                f"top_species.json lists {n_species} — regenerate the cache"
            )

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict:
        seq = str(self.seqs[idx])[: self.max_len]
        pooled = self.pooled[idx].astype(np.float32)
        mask = self.species_mask[idx]
        active = np.where(mask)[0]
        sp_idx = int(np.random.choice(active)) if len(active) > 0 \
            else int(np.random.randint(0, self.n_species))
        sp_onehot = np.zeros(self.n_species, dtype=np.float32)
        sp_onehot[sp_idx] = 1.0
        phys = compute_one(seq)
        charge_bucket = int(bucket_charge(float(phys[0])).item())
        length_bucket = int(bucket_length(min(len(seq), self.max_len)).item())
        return {"seq": seq, "pooled": pooled, "species_onehot": sp_onehot,
                "length_bucket": length_bucket, "charge_bucket": charge_bucket}


def make_collator(tokenizer, max_len: int = 100):
    def coll(batch):
        seqs = [b["seq"] for b in batch]
        enc = tokenizer(seqs, padding=True, truncation=True,
                        max_length=max_len + 2, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pooled": torch.tensor(np.stack([b["pooled"] for b in batch]), dtype=torch.float32),
            "species_onehot": torch.tensor(np.stack([b["species_onehot"] for b in batch]),
                                           dtype=torch.float32),
            "length_bucket": torch.tensor([b["length_bucket"] for b in batch], dtype=torch.long),
            "charge_bucket": torch.tensor([b["charge_bucket"] for b in batch], dtype=torch.long),
            "seqs": seqs,
        }
    return coll


def gen_loss(out: dict, batch: dict, *, beta: float, free_bits: float = 0.5,
             alpha_length: float = 0.1, alpha_recon: float = 0.5,
             pad_id: int = 0) -> tuple[torch.Tensor, dict]:
    input_ids = batch["input_ids"]
    lm_logits = out["lm_logits"][:, :-1, :].contiguous()
    tgt = input_ids[:, 1:].contiguous()
    lm_mask = tgt != pad_id
    if lm_mask.sum() == 0:
        l_ar = lm_logits.new_zeros(())
    else:
        l_ar = F.cross_entropy(lm_logits.reshape(-1, lm_logits.shape[-1]),
                               tgt.reshape(-1), ignore_index=pad_id)

    l_len = F.cross_entropy(out["length_logits"], batch["length_bucket"])
    l_recon = F.mse_loss(out["h_hat"], batch["pooled"])
    kl = kl_gaussian(out["mu_q"], out["log_sigma_q"], out["mu_p"], out["log_sigma_p"],
                     free_bits=free_bits).mean()

    total = l_ar + alpha_length * l_len + alpha_recon * l_recon + beta * kl
    return total, {"loss": total.detach(),
                   "loss_ar": l_ar.detach(),
                   "loss_length": l_len.detach(),
                   "loss_recon": l_recon.detach(),
                   "loss_kl": kl.detach(),
                   "beta": beta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache", default="data/current/latent_cache.npz")
    ap.add_argument("--data_dir", default="data/current")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out_dir or cfg.get("out_dir", "runs/stage3_gen"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the species list saved next to the cache; fall back to top_species.json.
    species_path = Path(args.cache).with_suffix(".species.json")
    if species_path.exists():
        species_list = json.load(open(species_path))["species_list"]
    else:
        species_list = load_latest_species(args.data_dir)
    n_species = len(species_list)

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"],
                                        cache_dir=cfg.get("hf_cache_dir"))
    ds = LatentCacheDataset(args.cache, tok, n_species=n_species,
                            max_len=cfg["data"].get("max_len", 100))
    coll = make_collator(tok, max_len=cfg["data"].get("max_len", 100))
    loader = DataLoader(ds, batch_size=cfg["train"].get("batch_size", 64),
                        shuffle=True, num_workers=cfg["train"].get("num_workers", 4),
                        collate_fn=coll, pin_memory=True, drop_last=True)
    print(f"latent cache: n={len(ds)}, n_species={n_species}, "
          f"batch={cfg['train']['batch_size']}, steps/epoch={len(loader)}")

    cvae_cfg = CVAEConfig(
        pooled_dim=cfg["model"].get("pooled_dim", 1280),
        z_dim=cfg["model"].get("z_dim", 128),
        hidden_dim=cfg["model"].get("hidden_dim", 512),
        n_species=n_species,
    )
    gen = Generator(tok, cvae_cfg,
                    ar_d_model=cfg["model"].get("ar_d_model", 512),
                    ar_n_layers=cfg["model"].get("ar_n_layers", 4)).cuda()
    n_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"generator trainable params: {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(gen.parameters(), lr=cfg["train"]["lr"],
                             weight_decay=cfg["train"].get("wd", 0.01))
    total_steps = args.max_steps or (cfg["train"]["epochs"] * len(loader))
    warmup = int(total_steps * cfg["train"].get("warmup_ratio", 0.05))
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_loss = float("inf")
    step = 0
    log_every = cfg["train"].get("log_every", 50)
    save_every = cfg["train"].get("save_every", 1000)
    beta_period = cfg["train"].get("beta_period", 10000)
    max_beta = cfg["train"].get("max_beta", 0.5)
    free_bits = cfg["train"].get("free_bits", 0.5)

    t0 = time.time()
    while step < total_steps:
        for batch in loader:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = gen(batch["pooled"], batch["input_ids"],
                      batch["species_onehot"], batch["length_bucket"], batch["charge_bucket"])
            beta = beta_schedule(step, period=beta_period, max_beta=max_beta)
            loss, parts = gen_loss(out, batch, beta=beta, free_bits=free_bits,
                                   pad_id=tok.pad_token_id)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg["train"].get("grad_clip", 1.0))
            opt.step()
            sched.step()
            step += 1
            if step % log_every == 0:
                dt = time.time() - t0
                msg = " ".join(f"{k[5:]}={float(v):.3f}"
                               for k, v in parts.items() if k.startswith("loss_"))
                print(f"[gen {step:6d}/{total_steps}] total={float(parts['loss']):.3f} {msg} "
                      f"β={beta:.3f} lr={sched.get_last_lr()[0]:.2e} {step/dt:.1f}step/s")
            if step % save_every == 0:
                if float(parts["loss"]) < best_loss:
                    best_loss = float(parts["loss"])
                    torch.save({"model": gen.state_dict(), "step": step, "loss": best_loss,
                                 "config": cfg, "vocab_size": tok.vocab_size,
                                 "species_list": species_list},
                               out_dir / "best.pt")
                    print(f"  saved best ({best_loss:.4f}) -> {out_dir/'best.pt'}")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    torch.save(gen.state_dict(), out_dir / "last.pt")
    print(f"done. best loss = {best_loss:.4f}")


if __name__ == "__main__":
    main()
