"""Genomic k-mer species features (for the LLAMP-style species encoder).

Produces a 340-d feature vector per species: concatenated mono/di/tri/tetra
nucleotide frequencies over `ACGT` (4 + 16 + 64 + 256). Each k-block is
normalised independently so block sums ≈ 1 (LLAMP convention).

Outputs:
  data/current/genomes/{species_slug}.fna.gz      reference genome (downloaded)
  data/current/species_kmer_features.npz          {names: [N], features: [N,340]}

Usage:
  python -m amp_prioritizer.species_genome build-features \
      --species_lists data/current/top_species.json data/current/holdout_species.json \
      --genome_dir data/current/genomes \
      --out data/current/species_kmer_features.npz

  python -m amp_prioritizer.species_genome download \
      --species "Streptococcus mutans" "Streptococcus sobrinus" \
      --out data/current/genomes
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

K_LIST: tuple[int, ...] = (1, 2, 3, 4)
TOTAL_DIM: int = sum(4 ** k for k in K_LIST)  # 4 + 16 + 64 + 256 = 340
_BASES: tuple[str, ...] = ("A", "C", "G", "T")


def species_slug(species_name: str) -> str:
    return species_name.strip().lower().replace(" ", "_").replace("/", "_")


def _kmer_index(k: int) -> dict[str, int]:
    """Lexicographic ACGT order → integer index within the k-block."""
    idx: dict[str, int] = {}
    stack: list[str] = [""]
    while stack:
        p = stack.pop(0)
        if len(p) == k:
            idx[p] = len(idx)
        else:
            for b in _BASES:
                stack.append(p + b)
    # the BFS above doesn't preserve order we want; rebuild deterministically
    ordered: list[str] = []

    def _gen(prefix: str, depth: int) -> None:
        if depth == 0:
            ordered.append(prefix)
            return
        for b in _BASES:
            _gen(prefix + b, depth - 1)

    _gen("", k)
    return {kmer: i for i, kmer in enumerate(ordered)}


_KMER_INDEX: dict[int, dict[str, int]] = {k: _kmer_index(k) for k in K_LIST}
_BLOCK_OFFSETS: dict[int, int] = {}
_off = 0
for _k in K_LIST:
    _BLOCK_OFFSETS[_k] = _off
    _off += 4 ** _k


# ------------------------------------------------------------------
# FASTA parsing
# ------------------------------------------------------------------
def _iter_fasta_records(path: Path) -> Iterator[str]:
    """Yield sequence chunks (one per contig), uppercase, whitespace-stripped."""
    opener = gzip.open if str(path).endswith(".gz") else open
    buf: list[str] = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                if buf:
                    yield "".join(buf).upper()
                    buf = []
            else:
                buf.append(line)
        if buf:
            yield "".join(buf).upper()


# ------------------------------------------------------------------
# k-mer counting
# ------------------------------------------------------------------
def compute_kmer_features(fasta_path: Path) -> np.ndarray:
    """Compute 340-d k-mer frequency vector (per-k-block normalised) from a FASTA."""
    counts = np.zeros(TOTAL_DIM, dtype=np.float64)
    block_totals = np.zeros(len(K_LIST), dtype=np.float64)

    for seq in _iter_fasta_records(Path(fasta_path)):
        # Split on any non-ACGT to avoid counting k-mers spanning N/IUPAC bases.
        run: list[str] = []
        current: list[str] = []
        for ch in seq:
            if ch in ("A", "C", "G", "T"):
                current.append(ch)
            else:
                if current:
                    run.append("".join(current))
                    current = []
        if current:
            run.append("".join(current))

        for chunk in run:
            L = len(chunk)
            for ki, k in enumerate(K_LIST):
                if L < k:
                    continue
                off = _BLOCK_OFFSETS[k]
                idx_map = _KMER_INDEX[k]
                for i in range(L - k + 1):
                    kmer = chunk[i : i + k]
                    j = idx_map.get(kmer)
                    if j is not None:
                        counts[off + j] += 1.0
                        block_totals[ki] += 1.0

    feats = np.zeros(TOTAL_DIM, dtype=np.float32)
    for ki, k in enumerate(K_LIST):
        off = _BLOCK_OFFSETS[k]
        size = 4 ** k
        tot = block_totals[ki]
        if tot > 0:
            feats[off : off + size] = (counts[off : off + size] / tot).astype(np.float32)
        # else: leave zeros (will fail the sanity assertion below)

    return feats


# ------------------------------------------------------------------
# Genome download (NCBI datasets CLI)
# ------------------------------------------------------------------
def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def download_genome(species_name: str, cache_dir: Path, overwrite: bool = False) -> Path:
    """Download a reference genome via `datasets` CLI, cache to {slug}.fna.gz.

    Returns the path to the cached FASTA. Raises if `datasets` is not on PATH.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = species_slug(species_name)
    out = cache_dir / f"{slug}.fna.gz"
    if out.exists() and not overwrite:
        return out

    if _which("datasets") is None:
        raise RuntimeError(
            "NCBI `datasets` CLI not on PATH. Install via scripts/setup_env.sh "
            "(downloads a static Linux binary)."
        )

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = tmp / f"{slug}.zip"
        cmd = [
            "datasets", "download", "genome", "taxon", species_name,
            "--reference", "--include", "genome",
            "--filename", str(zip_path),
        ]
        print(f"[download] {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"datasets download failed for {species_name!r}: {proc.stderr}"
            )

        # unzip and locate a *.fna file
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        fna_files = sorted(tmp.glob("ncbi_dataset/data/**/*.fna"))
        if not fna_files:
            fna_files = sorted(tmp.glob("**/*.fna"))
        if not fna_files:
            raise RuntimeError(f"no .fna extracted for {species_name!r}")
        src = fna_files[0]
        with open(src, "rb") as fin, gzip.open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"[download] wrote {out} ({out.stat().st_size/1e6:.1f} MB)")

    return out


# ------------------------------------------------------------------
# Feature table
# ------------------------------------------------------------------
def _load_species_lists(paths: Iterable[Path]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for p in paths:
        data = json.load(open(p))
        for sp in data.get("species_list", []):
            if sp not in seen:
                seen.add(sp)
                names.append(sp)
    return names


def build_feature_table(
    species_list: list[str],
    genome_dir: Path,
    out_npz: Path,
    auto_download: bool = False,
) -> None:
    genome_dir = Path(genome_dir)
    out_npz = Path(out_npz)
    feats: list[np.ndarray] = []
    for sp in species_list:
        slug = species_slug(sp)
        fasta = genome_dir / f"{slug}.fna.gz"
        if not fasta.exists():
            fasta_plain = genome_dir / f"{slug}.fna"
            if fasta_plain.exists():
                fasta = fasta_plain
            elif auto_download:
                fasta = download_genome(sp, genome_dir)
            else:
                raise FileNotFoundError(
                    f"missing genome for {sp!r}: expected {fasta} "
                    f"(or pass --auto_download)"
                )
        f = compute_kmer_features(fasta)
        # sanity: mono-nucleotide block sums to ~1
        mono_sum = float(f[_BLOCK_OFFSETS[1] : _BLOCK_OFFSETS[1] + 4].sum())
        assert abs(mono_sum - 1.0) < 1e-4 or mono_sum == 0.0, (
            f"{sp} mono-block sum {mono_sum} ≉ 1 (empty sequence?)"
        )
        print(f"[feat] {sp}: shape={f.shape} mono_sum={mono_sum:.4f}")
        feats.append(f)

    arr = np.stack(feats).astype(np.float32)
    assert arr.shape == (len(species_list), TOTAL_DIM), arr.shape
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, names=np.array(species_list, dtype=object), features=arr)
    print(f"[feat] wrote {out_npz} shape={arr.shape}")


def load_feature_table(
    npz_path: str | Path, species_order: list[str]
) -> np.ndarray:
    """Select rows from an npz in the given species order. Raises on missing."""
    data = np.load(npz_path, allow_pickle=True)
    names: list[str] = list(data["names"])
    features: np.ndarray = data["features"]
    name_to_idx = {n: i for i, n in enumerate(names)}
    rows = []
    for sp in species_order:
        if sp not in name_to_idx:
            raise KeyError(f"species {sp!r} missing from {npz_path}")
        rows.append(features[name_to_idx[sp]])
    return np.stack(rows).astype(np.float32)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _cli_download(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    for sp in args.species:
        download_genome(sp, out_dir, overwrite=args.overwrite)


def _cli_build(args: argparse.Namespace) -> None:
    if args.species_lists:
        names = _load_species_lists([Path(p) for p in args.species_lists])
    elif args.species:
        names = list(args.species)
    else:
        raise SystemExit("need --species_lists or --species")
    build_feature_table(names, Path(args.genome_dir), Path(args.out),
                        auto_download=args.auto_download)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_dl = sub.add_parser("download", help="download reference genomes via NCBI datasets CLI")
    ap_dl.add_argument("--species", nargs="+", required=True)
    ap_dl.add_argument("--out", default="data/current/genomes")
    ap_dl.add_argument("--overwrite", action="store_true")
    ap_dl.set_defaults(func=_cli_download)

    ap_bf = sub.add_parser("build-features", help="compute the 340-d feature table npz")
    ap_bf.add_argument("--species_lists", nargs="*",
                       help="one or more *.json with {'species_list': [...]}")
    ap_bf.add_argument("--species", nargs="*", help="explicit species names (alternative)")
    ap_bf.add_argument("--genome_dir", default="data/current/genomes")
    ap_bf.add_argument("--out", default="data/current/species_kmer_features.npz")
    ap_bf.add_argument("--auto_download", action="store_true")
    ap_bf.set_defaults(func=_cli_build)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
