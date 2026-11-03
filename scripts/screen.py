"""v3 inference / screening API.

Loads an AMPModel ckpt and scores candidate peptides. Supports two modes:
  - classic: rank by amp_score * sp/<target_species>
  - narrow : rank by NarrowScore (see amp_prioritizer/score.py)

CLI:
  python scripts/screen.py --mode narrow \
      --ckpt runs/full/best.pt --config configs/full.yaml \
      --fasta candidates.fasta \
      --target_species "Acinetobacter baumannii" \
      --offtarget_species "Escherichia coli" "Staphylococcus aureus" \
      --top_k 100 --out out.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from amp_prioritizer.model import AMPModel
from amp_prioritizer.schema import load_latest_species, sanitize_seq


def _load(cfg_path: str, ckpt_path: str, data_dir: str):
    cfg = yaml.safe_load(open(cfg_path))
    species_list = load_latest_species(data_dir)
    mcfg = cfg["model"]
    model = AMPModel.from_checkpoint(
        ckpt_path,
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
    )
    model.eval().cuda()
    thr_path = Path(ckpt_path).parent / "thresholds.json"
    thr = json.load(open(thr_path)) if thr_path.exists() else {}
    return model, cfg, thr, species_list


def _calibrate(p: np.ndarray, iso: dict | None) -> np.ndarray:
    if not iso or "isotonic_x" not in iso:
        return p
    xs = np.asarray(iso["isotonic_x"])
    ys = np.asarray(iso["isotonic_y"])
    return np.interp(p, xs, ys, left=ys[0], right=ys[-1])


@torch.no_grad()
def screen(
    seqs: list[str],
    *,
    ckpt: str, config: str,
    data_dir: str = "data/current",
    target_species: list[str] | str | None = None,
    offtarget_species: list[str] | None = None,
    mode: str = "classic",
    top_k: int = 100, batch: int = 32,
) -> pd.DataFrame:
    model, cfg, thr, species_list = _load(config, ckpt, data_dir)
    max_len = cfg["data"].get("max_len", 200)
    if isinstance(target_species, str):
        target_species = [target_species]

    valid_seqs = [s for s in (sanitize_seq(x) for x in seqs) if s is not None]

    rows: list[dict] = []
    for chunk_start in range(0, len(valid_seqs), batch):
        chunk = valid_seqs[chunk_start:chunk_start + batch]
        enc = model.tokenizer(chunk, padding=True, truncation=True,
                              max_length=max_len + 2, return_tensors="pt").to("cuda")
        out = model(enc["input_ids"], enc["attention_mask"])
        amp_p = torch.sigmoid(out["amp_logit"]).float().cpu().numpy()
        sp_p = torch.sigmoid(out["species_logit"]).float().cpu().numpy()
        mic_mu = out["mic_mu"].float().cpu().numpy()
        hemo_mu = out["hemo_mu"].float().cpu().numpy()

        for j, s in enumerate(chunk):
            row: dict = {"seq": s}
            row["amp_score"] = float(_calibrate(np.array([amp_p[j]]), thr.get("amp"))[0])
            for k, sp in enumerate(species_list):
                row[f"sp/{sp}"] = float(_calibrate(np.array([sp_p[j, k]]),
                                                   thr.get("species", {}).get(sp))[0])
                row[f"mic_mu/{sp}"] = float(mic_mu[j, k])
                row[f"mic_mu_cal/{sp}"] = float(_calibrate(np.array([mic_mu[j, k]]),
                                                            thr.get("mic", {}).get(sp))[0])
            row["hemo_mu"] = float(hemo_mu[j])
            rows.append(row)

    df = pd.DataFrame(rows)

    if mode == "narrow":
        if not target_species:
            raise ValueError("mode=narrow needs --target_species")
        from amp_prioritizer.score import narrow_spectrum_score
        df = narrow_spectrum_score(df, target_species, offtarget_species,
                                   species_list=species_list)
    else:
        if target_species:
            col = f"sp/{target_species[0]}"
            if col not in df.columns:
                raise ValueError(f"unknown species {target_species[0]}; valid: {species_list}")
            df["combined"] = df["amp_score"] * df[col]
            df = df.sort_values("combined", ascending=False).reset_index(drop=True)
        else:
            df = df.sort_values("amp_score", ascending=False).reset_index(drop=True)
    return df.head(top_k) if top_k else df


def _read_fasta(path: str) -> list[str]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", default="data/current")
    ap.add_argument("--fasta")
    ap.add_argument("--seqs", nargs="+")
    ap.add_argument("--mode", choices=["classic", "narrow"], default="classic")
    ap.add_argument("--target_species", nargs="+", default=None)
    ap.add_argument("--offtarget_species", nargs="+", default=None,
                    help="narrow mode only; None = all species \\ target")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--out", default="screen_out.csv")
    args = ap.parse_args()

    if args.fasta:
        seqs = _read_fasta(args.fasta)
    elif args.seqs:
        seqs = args.seqs
    else:
        sys.exit("provide --fasta or --seqs")

    df = screen(seqs, ckpt=args.ckpt, config=args.config, data_dir=args.data_dir,
                target_species=args.target_species,
                offtarget_species=args.offtarget_species,
                mode=args.mode, top_k=args.top_k)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()
