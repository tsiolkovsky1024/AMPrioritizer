#!/usr/bin/env python3
"""Download reference genomes for top_species + holdout_species via NCBI datasets CLI.

Usage:
  python scripts/download_genomes.py \
      --species_lists data/current/top_species.json data/current/holdout_species.json \
      --out data/current/genomes
  python scripts/download_genomes.py --extra "Streptococcus mutans" --out data/current/genomes

Builds the 340-d k-mer feature npz afterwards:
  python -m amp_prioritizer.species_genome build-features \
      --species_lists data/current/top_species.json data/current/holdout_species.json \
      --genome_dir data/current/genomes \
      --out data/current/species_kmer_features.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from amp_prioritizer.species_genome import download_genome, species_slug


def collect_species(species_lists: list[str], extra: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in species_lists or []:
        data = json.load(open(path))
        for sp in data.get("species_list", []):
            if sp not in seen:
                seen.add(sp)
                out.append(sp)
    for sp in extra or []:
        if sp not in seen:
            seen.add(sp)
            out.append(sp)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--species_lists", nargs="*", default=[],
                    help="paths to *.json files with {'species_list': [...]}")
    ap.add_argument("--extra", nargs="*", default=[], help="explicit species names")
    ap.add_argument("--out", default="data/current/genomes")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    species = collect_species(args.species_lists, args.extra)
    if not species:
        ap.error("nothing to download — pass --species_lists or --extra")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_skip = 0
    n_fail = 0
    for sp in species:
        slug = species_slug(sp)
        out_path = out_dir / f"{slug}.fna.gz"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {sp} -> {out_path} (exists)")
            n_skip += 1
            continue
        try:
            download_genome(sp, out_dir, overwrite=args.overwrite)
            n_done += 1
        except Exception as e:
            print(f"[FAIL] {sp}: {e}", file=sys.stderr)
            n_fail += 1
    print(f"summary: downloaded={n_done} skipped={n_skip} failed={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
