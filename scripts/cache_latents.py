"""scripts/cache_latents.py — encode AMP positives with frozen Stage-2 encoder into a .npz cache.

v3 only. Reads data/current/ via AMPMultiTaskDataset, dumps pooled ESM-2 embeddings
plus per-species MIC mask/log10_mic for the downstream Stage 3 CVAE + AR decoder.

Output schema (data/current/latent_cache.npz):
  seq          [N] object
  length       [N] int32
  pooled       [N, 1280] float32
  amp_label    [N] int8          (always 1 — we only cache positives)
  species_mask [N, K] bool       K = len(top_species_list)
  log10_mic    [N, K] float32    NaN where mask is False
  species_list stored in `_species_list.json` beside the cache

Launch:
  python scripts/cache_latents.py --ckpt runs/full/best.pt \
      --config configs/full.yaml --out data/current/latent_cache.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


from amp_prioritizer.schema import load_latest_species
from amp_prioritizer.dataset import AMPCollator, AMPMultiTaskDataset
from amp_prioritizer.model import AMPModel


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/current")
    ap.add_argument("--out", default="data/current/latent_cache.npz")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_len = cfg["data"].get("max_len", 200)
    species_list = load_latest_species(args.data_dir)
    mcfg = cfg["model"]

    model = AMPModel.from_checkpoint(
        args.ckpt,
        backbone_name=mcfg["name"],
        n_species=len(species_list),
        finetune_mode=mcfg.get("finetune_mode", "full"),
        lora_r=mcfg.get("lora_r", 16),
        lora_alpha=mcfg.get("lora_alpha", 32),
        lora_dropout=mcfg.get("lora_dropout", 0.05),
        head_hidden=mcfg.get("head_hidden", 512),
        head_dropout=mcfg.get("head_dropout", 0.2),
        gradient_checkpointing=False,
        hf_cache_dir=cfg.get("hf_cache_dir"),
        species_encoder_cfg=mcfg.get("species_encoder"),
        species_list=species_list,
    )
    model.eval().cuda()

    all_seq: list[str] = []
    all_pooled: list[np.ndarray] = []
    all_species_mask: list[np.ndarray] = []
    all_log10_mic: list[np.ndarray] = []
    physchem_cache_dir = Path(cfg["data"].get("physchem_cache_dir",
                                              f"{args.data_dir}/physchem"))

    for sp in args.splits:
        ds = AMPMultiTaskDataset(
            args.data_dir, split=sp, max_len=max_len,
            physchem_cache=physchem_cache_dir / f"{sp}.npz",
            active_threshold_log10=cfg["data"].get("active_threshold_log10", 1.5),
        )
        keep = np.where(ds.amp_label == 1)[0]
        if len(keep) == 0:
            print(f"[{sp}] no AMP positives, skipping")
            continue
        collator = AMPCollator(model.tokenizer, max_len=max_len + 2)
        sub = Subset(ds, keep.tolist())
        loader = DataLoader(sub, batch_size=args.batch, shuffle=False,
                            num_workers=2, collate_fn=collator, pin_memory=True)

        for batch in loader:
            out = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda())
            all_pooled.append(out["pooled"].float().cpu().numpy())

        for idx in keep:
            item = ds[int(idx)]
            all_seq.append(item["seq"])
            mask = item["mic_mask"].astype(bool)
            log10 = np.where(mask, item["mic_target"], np.nan).astype(np.float32)
            all_species_mask.append(mask)
            all_log10_mic.append(log10)
        print(f"[{sp}] encoded {len(keep)} sequences")

    pooled = np.concatenate(all_pooled, axis=0).astype(np.float32)
    species_mask = np.stack(all_species_mask, axis=0)
    log10_mic = np.stack(all_log10_mic, axis=0)
    length = np.array([len(s) for s in all_seq], dtype=np.int32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             seq=np.array(all_seq, dtype=object),
             length=length, pooled=pooled,
             species_mask=species_mask, log10_mic=log10_mic)
    json.dump({"species_list": species_list},
              open(out_path.with_suffix(".species.json"), "w"), indent=2)
    print(f"saved {len(all_seq)} sequences -> {out_path}")
    print(f"  pooled shape: {pooled.shape}, dtype={pooled.dtype}")
    print(f"  species_mask cells with MIC: {int(species_mask.sum())}")
    print(f"  species_list -> {out_path.with_suffix('.species.json')}")


if __name__ == "__main__":
    main()
