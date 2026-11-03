"""Standardized AMP data schema.

Fields per row (parquet):
  seq:               str   raw sequence (single-letter amino acids, uppercase, only the 20 standard + X/B/Z/U/O)
  length:            int   sequence length
  amp_label:         int   1 = AMP, 0 = non-AMP, -1 = unlabeled
  species_taxid:     int   NCBI Taxonomy ID, -1 = unspecified / general AMP
  species_group:     str   normalized species group (ESKAPE-Ec, ESKAPE-Sa, ESKAPE-Kp, ESKAPE-Ab,
                          ESKAPE-Pa, ESKAPE-Ef, EColi, SAureus, OTHER, NA)
  active_label:      int   whether the peptide is active against this species (1=active, 0=inactive, -1=unlabeled)
  mic_value:         float MIC value (μM or μg/mL)
  mic_unit:          str   "uM" | "ug/mL" | ""
  source:            str   data source label (zswitten, camera, fuseAMP, abaumannii_few_shot,
                          huang_pipeline, dbAMP, DBAASP, ...)
  source_priority:   int   confidence ordering, 1 = highest
  conflict_group_id: int   MMseqs2 cluster id, used to detect duplicates/conflicts
  split:             str   train | val | test | unassigned
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import pyarrow as pa


# ESKAPE species plus the default Ec/Sa groups (7 species heads)
ESKAPE_SPECIES = [
    ("Enterococcus faecium",       1352),
    ("Staphylococcus aureus",      1280),
    ("Klebsiella pneumoniae",      573),
    ("Acinetobacter baumannii",    470),
    ("Pseudomonas aeruginosa",     287),
    ("Enterobacter cloacae",       550),  # 'E' in ESKAPE
    ("Escherichia coli",           562),
]
SPECIES_GROUPS = [name for name, _ in ESKAPE_SPECIES]  # 7 fixed heads
SPECIES_TO_INDEX = {name: i for i, name in enumerate(SPECIES_GROUPS)}
TAXID_TO_GROUP = {tid: name for name, tid in ESKAPE_SPECIES}

# Source confidence (lower number = more reliable)
SOURCE_PRIORITY = {
    "DBAASP":                 1,
    "dbAMP":                  2,
    "APD3":                   3,
    "huang_pipeline":         4,   # Wang et al 2024 ML-mined AMPs
    "camera":                 5,
    "abaumannii_few_shot":    6,
    "fuseAMP":                7,
    "zswitten":               8,
    "literature":             9,
    "synthetic_negative":     10,
    "shuffled_negative":      11,
}


SCHEMA = pa.schema([
    ("seq",                pa.string()),
    ("length",             pa.int32()),
    ("amp_label",          pa.int8()),
    ("species_taxid",      pa.int32()),
    ("species_group",      pa.string()),
    ("active_label",       pa.int8()),
    ("mic_value",          pa.float32()),
    ("mic_unit",           pa.string()),
    ("source",             pa.string()),
    ("source_priority",    pa.int16()),
    ("conflict_group_id",  pa.int32()),
    ("split",              pa.string()),
])


@dataclass
class AMPRecord:
    seq: str
    amp_label: int = -1
    species_taxid: int = -1
    species_group: str = "NA"
    active_label: int = -1
    mic_value: float = float("nan")
    mic_unit: str = ""
    source: str = ""
    source_priority: int = 99
    conflict_group_id: int = -1
    split: str = "unassigned"

    @property
    def length(self) -> int:
        return len(self.seq)


def empty_frame() -> pd.DataFrame:
    cols = [f.name for f in SCHEMA]
    return pd.DataFrame({c: pd.Series(dtype=str(SCHEMA.field(c).type)) for c in cols})


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBZUO")


# === V2 (narrow-spectrum pipeline) additions ===
# Right-censored MIC sentinel: rows with `mic_value == SENTINEL_MIC_UM` represent
# experiments where the peptide was inactive up to the highest tested concentration
# (≈ 2¹³ µM). Fitting these literally with Huber/MSE biases the regressor toward
# this wall value; we treat them as right-censored under a Tobit/Gaussian-NLL likelihood.
SENTINEL_MIC_UM: float = 8195.997
SENTINEL_LOG10_UM: float = 3.91365  # log10(SENTINEL_MIC_UM)


def is_sentinel(mic_value: float, tol: float = 1e-3) -> bool:
    """True if `mic_value` (linear µM) is the inactive-sentinel marker."""
    if mic_value is None:
        return False
    try:
        v = float(mic_value)
    except (TypeError, ValueError):
        return False
    return abs(v - SENTINEL_MIC_UM) < tol


# Auxiliary physicochemical regression targets (Stage 2a anchor signals).
# Deterministically computable from `seq` via BioPython — see `amp_prioritizer/physchem.py`.
PHYSCHEM_FIELDS: list[str] = [
    "net_charge",            # net charge at pH 7.4 (Henderson-Hasselbalch)
    "gravy",                 # Kyte-Doolittle hydropathy mean
    "hydrophobic_moment",    # Eisenberg amphipathic moment, helix window 11
    "helix_propensity",      # Chou-Fasman α-helix tendency, mean over residues
    "isoelectric_point",     # pI via BioPython ProteinAnalysis
]


# === V3 (Latest dataset) additions ===
# data/current/ is produced by amp_prioritizer/ingest.py from the upstream integrated snapshot.
# The species list is dynamic (top-K) and exposed via data/current/top_species.json.

TARGET_GROUPS: list[str] = ["Gram+", "Gram-", "Fungus", "Virus", "Parasite"]
TARGET_GROUP_TO_INDEX = {g: i for i, g in enumerate(TARGET_GROUPS)}


def load_latest_species(data_dir: str = "data/current") -> list[str]:
    """Read data/current/top_species.json. If the file is missing, fall back to SPECIES_GROUPS (the legacy 7 species)."""
    import json
    from pathlib import Path
    p = Path(data_dir) / "top_species.json"
    if not p.exists():
        return list(SPECIES_GROUPS)
    return list(json.load(open(p))["species_list"])


# New censoring semantics: there is no longer a "fixed sentinel value"; each row carries its own
# (mic_value, right/left censored flag).
# parse_operator_flags converts the 'operator_flags' field of mic_summary.csv (a CSV string like "=,>")
# into a (right_censored, left_censored) tuple.
def parse_operator_flags(flags: str) -> tuple[bool, bool]:
    """'=' / 'range' → (False, False); '>' or '>=' present → right=True;
    '<' or '<=' present → left=True. Both sides present → both True (rare)."""
    if not flags:
        return (False, False)
    parts = {p.strip() for p in str(flags).split(",")}
    right = bool(parts & {">", ">="})
    left = bool(parts & {"<", "<="})
    return (right, left)


def sanitize_seq(s: str) -> Optional[str]:
    """Uppercase, strip whitespace; reject sequences with non-AA chars or length out of [3, 200]."""
    if not isinstance(s, str):
        return None
    s = "".join(s.split()).upper()
    if not s or len(s) < 3 or len(s) > 200:
        return None
    if any(ch not in VALID_AA for ch in s):
        return None
    return s
