"""One-shot: merge runs/stage0_mlm/best.pt LoRA adapters into ESM-2 backbone.

Stage 0 (amp_prioritizer/mlm_pretrain.py) wraps ESM-2 t33 650M with PEFT LoRA and saves a flat
state_dict to runs/stage0_mlm/best.pt. To use it as a warm-start for full-FT in our
compact Stage 2 trainer we need a plain HF directory with the LoRA deltas merged
into the backbone weights.

Usage:
    python -m amp_prioritizer.merge_stage0_lora \
        --ckpt runs/stage0_mlm/best.pt \
        --out  runs/stage0_merged

Re-running is safe; the destination directory is overwritten.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Use a project-local HF cache to avoid re-downloading the backbone weights.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage0_mlm/best.pt",
                    help="Stage 0 best ckpt (state_dict torch.save format).")
    ap.add_argument("--stage0_config", default="configs/stage0_mlm.yaml",
                    help="Used to read the LoRA cfg + backbone name.")
    ap.add_argument("--out", default="runs/stage0_merged",
                    help="Destination HF model directory.")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(ROOT / args.stage0_config))
    bb_name = cfg["model"]["name"]
    lora_r = cfg["model"]["lora_r"]
    lora_alpha = cfg["model"]["lora_alpha"]
    lora_dropout = cfg["model"]["lora_dropout"]

    ckpt_path = (ROOT / args.ckpt) if not Path(args.ckpt).is_absolute() else Path(args.ckpt)
    out_dir = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        sys.exit(f"[merge] checkpoint not found: {ckpt_path}")

    print(f"[merge] loading base backbone: {bb_name}", flush=True)
    base = AutoModelForMaskedLM.from_pretrained(bb_name, cache_dir=str(ROOT / ".hf_cache"))
    tok  = AutoTokenizer.from_pretrained(bb_name, cache_dir=str(ROOT / ".hf_cache"))

    print(f"[merge] wrapping with LoRA cfg "
          f"(r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})", flush=True)
    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="FEATURE_EXTRACTION",
        target_modules=["query", "key", "value", "dense"],
    )
    peft_model = get_peft_model(base, lora_cfg)

    print(f"[merge] loading state_dict from {ckpt_path}", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)
    missing, unexpected = peft_model.load_state_dict(sd, strict=False)
    print(f"[merge] missing={len(missing)}  unexpected={len(unexpected)}", flush=True)
    if missing[:3]:
        print(f"  e.g. missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"  e.g. unexpected: {unexpected[:3]}")

    print("[merge] merge_and_unload()", flush=True)
    merged = peft_model.merge_and_unload()

    print(f"[merge] saving plain HF model to {out_dir}", flush=True)
    merged.base_model.save_pretrained(str(out_dir)) if hasattr(merged, "base_model") \
        else merged.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"[merge] done. point model.name in the training config -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
