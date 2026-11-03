"""Stage 2 auxiliary supervision: sequence → 5-dim physicochemical vector.

PHYSCHEM_FIELDS (amp_prioritizer/schema.py):
  0. net_charge          — net charge at pH 7.4 (Henderson-Hasselbalch)
  1. gravy               — mean Kyte-Doolittle hydropathy
  2. hydrophobic_moment  — Eisenberg amphipathic moment (helix 100°, window=11, max over sliding windows)
  3. helix_propensity    — mean Chou-Fasman α-helix propensity
  4. isoelectric_point   — pI (BioPython ProteinAnalysis)

Design rationale:
  - All 5 values are deterministic closed-form functions of the sequence, so they are computed once and cached.
  - As a Stage 2 *auxiliary regression target*: even when MIC is missing, the model is forced to
    encode these physicochemical properties in its embedding — robustness to noise / prevents simply memorizing MIC labels.
  - No external downloads required, pure BioPython.

API:
  compute_one(seq) -> np.ndarray(5)
  compute_many(seqs, cache_path=None) -> np.ndarray(N, 5)   (optional disk cache)
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd  # Kyte-Doolittle dict


# Chou-Fasman α-helix propensity (Pα) — standard table (Chou & Fasman 1978).
# Non-canonical (B, Z, X, U, O) get 1.0 (neutral).
CHOU_FASMAN_ALPHA = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "E": 1.51, "Q": 1.11, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}

# Map non-standard residues to the nearest standard residue so BioPython does not raise.
# B = Asx (D/N), Z = Glx (E/Q), U = selenocysteine -> C, O = pyrrolysine -> K, X = unknown -> A.
NONSTANDARD_REPLACE = {"B": "D", "Z": "E", "U": "C", "O": "K", "X": "A"}


def _normalize(seq: str) -> str:
    return "".join(NONSTANDARD_REPLACE.get(c, c) for c in seq.upper())


def _hydrophobic_moment(seq: str, window: int = 11, angle_deg: float = 100.0) -> float:
    """Eisenberg amphipathic moment, sliding max over windows of length `window`.

    μH = | Σ_i h_i * (cos(iθ), sin(iθ)) | / window
    where h_i is the Kyte-Doolittle hydropathy, θ = 100° (α-helix turn).
    Short sequences (< window) are computed once over the whole sequence.
    """
    seq = _normalize(seq)
    h = np.array([kd.get(c, 0.0) for c in seq], dtype=np.float64)
    n = h.size
    if n == 0:
        return 0.0
    angle = math.radians(angle_deg)
    w = min(window, n)
    best = 0.0
    for start in range(0, n - w + 1):
        sub = h[start:start + w]
        idx = np.arange(w)
        cos_s = float(np.sum(sub * np.cos(idx * angle)))
        sin_s = float(np.sum(sub * np.sin(idx * angle)))
        m = math.hypot(cos_s, sin_s) / w
        if m > best:
            best = m
    if w == n and best == 0.0:  # sliding window never ran → single window
        idx = np.arange(n)
        cos_s = float(np.sum(h * np.cos(idx * angle)))
        sin_s = float(np.sum(h * np.sin(idx * angle)))
        best = math.hypot(cos_s, sin_s) / n
    return best


def _helix_propensity(seq: str) -> float:
    seq = _normalize(seq)
    if not seq:
        return 1.0
    vals = [CHOU_FASMAN_ALPHA.get(c, 1.0) for c in seq]
    return float(sum(vals) / len(vals))


def compute_one(seq: str) -> np.ndarray:
    """5-dim physchem vector, ordered to match PHYSCHEM_FIELDS."""
    norm = _normalize(seq)
    pa = ProteinAnalysis(norm)
    try:
        charge = float(pa.charge_at_pH(7.4))
    except Exception:
        charge = 0.0
    try:
        gravy = float(pa.gravy())
    except Exception:
        gravy = 0.0
    try:
        pi = float(pa.isoelectric_point())
    except Exception:
        pi = 7.0
    moment = _hydrophobic_moment(norm)
    helix = _helix_propensity(norm)
    return np.array([charge, gravy, moment, helix, pi], dtype=np.float32)


def _hash_seqs(seqs: Iterable[str]) -> str:
    h = hashlib.sha1()
    for s in seqs:
        h.update(s.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def compute_many(seqs: list[str], cache_path: str | Path | None = None) -> np.ndarray:
    """Batch compute. If cache_path is given and the file exists with a matching hash, load it directly.

    The cache file is an .npz with keys: vec [N,5], hash (str).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                data = np.load(cache_path, allow_pickle=True)
                cached_hash = str(data["hash"]) if "hash" in data.files else ""
                if cached_hash == _hash_seqs(seqs):
                    return data["vec"]
            except Exception:
                pass  # fall through to recompute

    vec = np.stack([compute_one(s) for s in seqs], axis=0).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, vec=vec, hash=_hash_seqs(seqs))
    return vec


if __name__ == "__main__":
    import sys
    # Smoke test
    examples = [
        "GIGAVLKVLTTGLPALISWIKRKRQQ",   # Magainin-2 (cationic α-helical AMP)
        "MKLLVLALLLVAALA",              # Hydrophobic signal-peptide-like
        "DDDDDDDDDDDDD",                # Acidic peptide (negative net charge)
        "KKKKKKKKKKKKK",                # Highly cationic
    ]
    arr = compute_many(examples)
    print("seq | charge | gravy | moment | helix | pI")
    for s, v in zip(examples, arr):
        print(f"{s[:24]:24s} | {v[0]:+6.2f} | {v[1]:+5.2f} | {v[2]:5.3f} | {v[3]:.3f} | {v[4]:5.2f}")
    sys.exit(0)
