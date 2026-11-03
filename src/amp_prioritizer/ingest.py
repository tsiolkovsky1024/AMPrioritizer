"""Build data/current/*.parquet from an integrated source snapshot.

Inputs:
  <integrated>/peptides.jsonl
  <integrated>/activities.jsonl
  <integrated>/mic_summary.csv   (used only for cross-checking)

Outputs (data/current/):
  peptides.parquet     (dedup_key, seq, length, n_sources, has_modification, source_list)
  mic_long.parquet     (dedup_key, target_species, target_group, mic_uM_geomean, log10_mic,
                        censored_right, censored_left, n_measurements)
  mic_wide.parquet     (dedup_key + per-top-K-species columns: mic/log10_mic/censored_right/_left)
  hemolysis.parquet    (dedup_key, hc50_uM_geomean, log10_hc50, censored_right, censored_left, n_measurements)
  splits.parquet       (dedup_key, split ∈ {train,val,test})
  top_species.json     {"species_list": [...], "n_rows_per_species": {...}}

Constraints:
  - Species are keyed by target_species_normalized (provided by activities.jsonl). target_species_raw in
    mic_summary.csv contains the strain (ATCC xxx), so the same species with different strains would duplicate;
    here the data is re-aggregated by normalized species.
  - top-K species defaults to K=20, ordered by (n_measurements), but the 7 ESKAPE+Ec species are always included.
  - operator_flags parsing: '>' or '>=' → right-censored; '<' or '<=' → left-censored. Both present → both flagged True.
  - Geomean: exp(mean(log(value_uM)))
  - Sequence normalization: sequence_canonical, length ∈ [3, 200], only the 20 standard AA + X/B/Z/U/O.

Usage:
  python -m amp_prioritizer.ingest --integrated /path/to/integrated \\
                               --out data/current --top_k 20
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from amp_prioritizer.schema import sanitize_seq


# ESKAPE + Ec — species that are always included
GUARANTEED_SPECIES = [
    "Enterococcus faecium",
    "Staphylococcus aureus",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Enterobacter cloacae",
    "Escherichia coli",
]
TARGET_GROUPS = ["Gram+", "Gram-", "Fungus", "Virus", "Parasite"]


# The upstream target_species_normalized still keeps both "S. aureus" and "Staphylococcus aureus" spellings.
# Here the most common abbreviations are merged manually so the top-K list contains no
# "two spellings of the same species".
SPECIES_ABBREV = {
    "S. aureus":       "Staphylococcus aureus",
    "E. coli":         "Escherichia coli",
    "P. aeruginosa":   "Pseudomonas aeruginosa",
    "K. pneumoniae":   "Klebsiella pneumoniae",
    "A. baumannii":    "Acinetobacter baumannii",
    "E. faecium":      "Enterococcus faecium",
    "E. faecalis":     "Enterococcus faecalis",
    "C. albicans":     "Candida albicans",
    "B. subtilis":     "Bacillus subtilis",
    "S. epidermidis":  "Staphylococcus epidermidis",
    "S. enterica":     "Salmonella enterica",
    "L. monocytogenes":"Listeria monocytogenes",
    "M. tuberculosis": "Mycobacterium tuberculosis",
    "M. luteus":       "Micrococcus luteus",
    "M. smegmatis":    "Mycobacterium smegmatis",
    "B. cereus":       "Bacillus cereus",
}


def canonicalize_species(s: str | None) -> str | None:
    if not s:
        return None
    s = str(s).strip()
    if s in SPECIES_ABBREV:
        return SPECIES_ABBREV[s]
    return s


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_operator(op: str | None) -> tuple[bool, bool]:
    """→ (right_censored, left_censored). '=' / 'range' / '' → (False, False)."""
    if not op:
        return (False, False)
    op = str(op).strip()
    if op in (">", ">="):
        return (True, False)
    if op in ("<", "<="):
        return (False, True)
    return (False, False)


def aggregate_censored_values(
    values: np.ndarray,
    right: np.ndarray,
    left: np.ndarray,
) -> dict | None:
    """Aggregate repeated observations of one latent endpoint.

    Exact replicates use their geometric mean. One-sided censored replicates
    retain the strongest bound. Mixed observation types cannot be represented
    by the current exact/one-sided row schema and are excluded explicitly.
    """
    values = np.asarray(values, dtype=float)
    right = np.asarray(right, dtype=bool)
    left = np.asarray(left, dtype=bool)
    if not (len(values) == len(right) == len(left)) or len(values) == 0:
        raise ValueError("values and censoring flags must be non-empty and aligned")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("censored endpoint values must be finite and positive")
    if np.logical_and(right, left).any():
        raise ValueError("one measurement cannot be both right- and left-censored")

    exact = ~(right | left)
    counts = {
        "n_exact": int(exact.sum()),
        "n_right": int(right.sum()),
        "n_left": int(left.sum()),
    }
    present = [name for name, count in counts.items() if count]
    if len(present) != 1:
        return None

    if counts["n_exact"]:
        representative = float(np.exp(np.log(values).mean()))
        observation_type = "exact"
        aggregation_method = "geometric_mean"
    elif counts["n_right"]:
        representative = float(values.max())
        observation_type = "right"
        aggregation_method = "maximum_lower_bound"
    else:
        representative = float(values.min())
        observation_type = "left"
        aggregation_method = "minimum_upper_bound"

    return {
        "value_uM": representative,
        "log10_value": float(math.log10(representative)),
        "censored_right": observation_type == "right",
        "censored_left": observation_type == "left",
        "observation_type": observation_type,
        "aggregation_method": aggregation_method,
        "n_measurements": int(len(values)),
        **counts,
    }


def _mixed_signature(grp: pd.DataFrame) -> str:
    exact = ~(grp["right"].astype(bool) | grp["left"].astype(bool))
    labels = []
    if exact.any():
        labels.append("exact")
    if grp["right"].any():
        labels.append("right")
    if grp["left"].any():
        labels.append("left")
    return "+".join(labels)


def build_peptides(integrated_dir: Path) -> pd.DataFrame:
    """Same dedup_key appearing under different sources → keep the longest canonical seq or raw seq."""
    by_key: dict[str, dict] = {}
    n_rows = 0
    for rec in iter_jsonl(integrated_dir / "peptides.jsonl"):
        n_rows += 1
        key = rec.get("dedup_key")
        if not key:
            continue
        seq = rec.get("sequence_canonical") or rec.get("sequence_raw") or ""
        seq = sanitize_seq(seq)
        if not seq:
            continue
        bucket = by_key.setdefault(key, {
            "dedup_key": key,
            "seq": seq, "length": len(seq),
            "source_list": [], "has_modification": False,
            "n_sources": 0,
        })
        # keep the longest (most informative) seq
        if len(seq) > bucket["length"]:
            bucket["seq"] = seq
            bucket["length"] = len(seq)
        src = rec.get("source_dataset")
        if src and src not in bucket["source_list"]:
            bucket["source_list"].append(src)
            bucket["n_sources"] = len(bucket["source_list"])
        if rec.get("nterm_modification") or rec.get("cterm_modification"):
            bucket["has_modification"] = True

    df = pd.DataFrame(list(by_key.values()))
    df["source_list"] = df["source_list"].apply(lambda lst: ",".join(sorted(lst)))
    print(f"peptides.jsonl: {n_rows} rows → {len(df)} unique sanitized peptides")
    return df


def build_mic_long(
    integrated_dir: Path,
    valid_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build one valid exact or one-sided row per peptide/species unit."""
    rows: list[dict] = []
    skipped_no_value = 0
    skipped_no_species = 0
    skipped_no_key = 0
    for rec in iter_jsonl(integrated_dir / "activities.jsonl"):
        if rec.get("endpoint") != "MIC":
            continue
        v = rec.get("value_uM")
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
            skipped_no_value += 1
            continue
        species = rec.get("target_species_normalized")
        species = canonicalize_species(species)
        if not species:
            skipped_no_species += 1
            continue
        key = rec.get("source_id")  # source_id from the peptides table, not dedup_key
        if not key:
            skipped_no_key += 1
            continue
        right, left = parse_operator(rec.get("operator"))
        rows.append({
            "source_dataset": rec.get("source_dataset"),
            "source_id": key,
            "target_species": species,
            "target_group": rec.get("target_group") or "",
            "value_uM": float(v),
            "right": right, "left": left,
        })
    print(f"activities MIC rows: kept {len(rows)}; "
          f"skipped no_value={skipped_no_value} no_species={skipped_no_species} no_key={skipped_no_key}")

    df = pd.DataFrame(rows)
    # join in dedup_key (from peptides.jsonl)
    src_to_key: dict[tuple[str, str], str] = {}
    for rec in iter_jsonl(integrated_dir / "peptides.jsonl"):
        sid = rec.get("source_id")
        sd = rec.get("source_dataset")
        dk = rec.get("dedup_key")
        if sid and sd and dk:
            src_to_key[(sd, sid)] = dk
    df["dedup_key"] = [src_to_key.get((sd, sid)) for sd, sid in zip(df["source_dataset"], df["source_id"])]
    df = df[df["dedup_key"].notna()].reset_index(drop=True)
    print(f"after join dedup_key: {len(df)}")

    records: list[dict] = []
    mixed_groups: Counter[str] = Counter()
    mixed_measurements: Counter[str] = Counter()
    grouped = df.groupby(["dedup_key", "target_species"], sort=False)
    for (dedup_key, species), grp in grouped:
        result = aggregate_censored_values(
            grp["value_uM"].to_numpy(),
            grp["right"].to_numpy(),
            grp["left"].to_numpy(),
        )
        if result is None:
            signature = _mixed_signature(grp)
            mixed_groups[signature] += 1
            mixed_measurements[signature] += len(grp)
            continue
        target_groups = grp.loc[grp["target_group"].astype(bool), "target_group"]
        records.append({
            "dedup_key": dedup_key,
            "target_species": species,
            "target_group": target_groups.mode().iloc[0] if len(target_groups) else "",
            "mic_uM_geomean": result.pop("value_uM"),
            "log10_mic": result.pop("log10_value"),
            **result,
        })

    agg_df = pd.DataFrame(records)
    excluded_orphan_groups = 0
    if valid_keys is not None:
        orphan = ~agg_df["dedup_key"].isin(valid_keys)
        excluded_orphan_groups = int(orphan.sum())
        agg_df = agg_df.loc[~orphan].reset_index(drop=True)
    source_counts = df["source_dataset"].fillna("unknown").value_counts()
    audit = {
        "policy": {
            "unit": "dedup_key + canonical target species",
            "exact": "geometric mean",
            "right_censored": "maximum lower bound",
            "left_censored": "minimum upper bound",
            "mixed_observation_types": "exclude entire unit",
        },
        "input_measurements": int(len(df)),
        "input_groups": int(grouped.ngroups),
        "retained_groups": int(len(agg_df)),
        "excluded_mixed_groups": int(sum(mixed_groups.values())),
        "excluded_mixed_measurements": int(sum(mixed_measurements.values())),
        "excluded_orphan_groups": excluded_orphan_groups,
        "mixed_groups_by_signature": dict(sorted(mixed_groups.items())),
        "mixed_measurements_by_signature": dict(sorted(mixed_measurements.items())),
        "retained_groups_by_type": {
            key: int(value)
            for key, value in agg_df["observation_type"].value_counts().items()
        },
        "measurements_by_source": {
            key: int(value) for key, value in source_counts.items()
        },
    }
    print(
        "mic_long censor-safe aggregation: "
        f"retained={len(agg_df)} excluded_mixed={audit['excluded_mixed_groups']}"
    )
    return agg_df, audit


def pick_top_species(mic_long: pd.DataFrame, k: int,
                     exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    counts = mic_long["target_species"].value_counts()
    candidates = [sp for sp in counts.index if sp and sp not in exclude]
    # ensure the 7 ESKAPE+Ec species are included
    forced = [sp for sp in GUARANTEED_SPECIES if sp in counts.index and sp not in exclude]
    remainder = [sp for sp in candidates if sp not in forced]
    final = forced + remainder
    final = final[:k]
    # but the forced species must stay in the list
    for f in forced:
        if f not in final:
            final = final[:-1] + [f]
    return final


def build_mic_wide(mic_long: pd.DataFrame, top_species: list[str]) -> pd.DataFrame:
    """Long → wide: one column per species, with mic/log10/censored_r/censored_l."""
    sub = mic_long[mic_long["target_species"].isin(top_species)].copy()
    # use pivot_table; multi-column output
    keys = sub["dedup_key"].unique()
    sp_to_idx = {sp: i for i, sp in enumerate(top_species)}
    N = len(top_species)

    mic_arr = np.full((len(keys), N), np.nan, dtype=np.float32)
    log10_arr = np.full((len(keys), N), np.nan, dtype=np.float32)
    cr_arr = np.zeros((len(keys), N), dtype=np.bool_)
    cl_arr = np.zeros((len(keys), N), dtype=np.bool_)
    n_arr = np.zeros((len(keys), N), dtype=np.int32)

    key_to_idx = {k: i for i, k in enumerate(keys)}
    for _, row in sub.iterrows():
        i = key_to_idx[row["dedup_key"]]
        j = sp_to_idx[row["target_species"]]
        mic_arr[i, j] = row["mic_uM_geomean"]
        log10_arr[i, j] = row["log10_mic"]
        cr_arr[i, j] = bool(row["censored_right"])
        cl_arr[i, j] = bool(row["censored_left"])
        n_arr[i, j] = int(row["n_measurements"])

    df = pd.DataFrame({"dedup_key": keys})
    for j, sp in enumerate(top_species):
        df[f"mic_uM/{sp}"] = mic_arr[:, j]
        df[f"log10_mic/{sp}"] = log10_arr[:, j]
        df[f"censored_right/{sp}"] = cr_arr[:, j]
        df[f"censored_left/{sp}"] = cl_arr[:, j]
        df[f"n_measurements/{sp}"] = n_arr[:, j]
    return df


def build_hemolysis(
    integrated_dir: Path,
    valid_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate hemolysis with the same censor-safe policy as MIC."""
    rows: list[dict] = []
    for rec in iter_jsonl(integrated_dir / "activities.jsonl"):
        if rec.get("endpoint") != "Hemolysis":
            continue
        v = rec.get("value_uM")
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
            continue
        right, left = parse_operator(rec.get("operator"))
        rows.append({
            "source_dataset": rec.get("source_dataset"),
            "source_id": rec.get("source_id"),
            "value_uM": float(v),
            "right": right, "left": left,
        })
    print(f"hemolysis rows kept: {len(rows)}")
    if not rows:
        empty = pd.DataFrame(columns=[
            "dedup_key", "hc50_uM_geomean", "log10_hc50",
            "censored_right", "censored_left", "observation_type",
            "aggregation_method", "n_measurements", "n_exact", "n_right", "n_left",
        ])
        return empty, {
            "input_measurements": 0,
            "input_groups": 0,
            "retained_groups": 0,
            "excluded_mixed_groups": 0,
        }

    df = pd.DataFrame(rows)
    # join dedup_key
    src_to_key: dict[tuple[str, str], str] = {}
    for rec in iter_jsonl(integrated_dir / "peptides.jsonl"):
        sid = rec.get("source_id"); sd = rec.get("source_dataset"); dk = rec.get("dedup_key")
        if sid and sd and dk:
            src_to_key[(sd, sid)] = dk
    df["dedup_key"] = [src_to_key.get((sd, sid)) for sd, sid in zip(df["source_dataset"], df["source_id"])]
    df = df[df["dedup_key"].notna()].reset_index(drop=True)

    records: list[dict] = []
    mixed_groups: Counter[str] = Counter()
    grouped = df.groupby("dedup_key", sort=False)
    for dedup_key, grp in grouped:
        result = aggregate_censored_values(
            grp["value_uM"].to_numpy(),
            grp["right"].to_numpy(),
            grp["left"].to_numpy(),
        )
        if result is None:
            mixed_groups[_mixed_signature(grp)] += 1
            continue
        records.append({
            "dedup_key": dedup_key,
            "hc50_uM_geomean": result.pop("value_uM"),
            "log10_hc50": result.pop("log10_value"),
            **result,
        })
    out = pd.DataFrame(records)
    excluded_orphan_groups = 0
    if valid_keys is not None:
        orphan = ~out["dedup_key"].isin(valid_keys)
        excluded_orphan_groups = int(orphan.sum())
        out = out.loc[~orphan].reset_index(drop=True)
    audit = {
        "policy": "same as MIC",
        "input_measurements": int(len(df)),
        "input_groups": int(grouped.ngroups),
        "retained_groups": int(len(out)),
        "excluded_mixed_groups": int(sum(mixed_groups.values())),
        "excluded_orphan_groups": excluded_orphan_groups,
        "mixed_groups_by_signature": dict(sorted(mixed_groups.items())),
        "retained_groups_by_type": {
            key: int(value)
            for key, value in out["observation_type"].value_counts().items()
        },
    }
    print(
        "hemolysis censor-safe aggregation: "
        f"retained={len(out)} excluded_mixed={audit['excluded_mixed_groups']}"
    )
    return out, audit


def build_splits(peptides: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(peptides)
    perm = rng.permutation(n)
    t_cut = int(n * 0.8)
    v_cut = int(n * 0.9)
    split = np.empty(n, dtype=object)
    split[perm[:t_cut]] = "train"
    split[perm[t_cut:v_cut]] = "val"
    split[perm[v_cut:]] = "test"
    return pd.DataFrame({"dedup_key": peptides["dedup_key"].values, "split": split})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated", required=True, help="directory with the integrated source snapshot")
    ap.add_argument("--out", default="data/current")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument(
        "--top_species_file",
        help="JSON file whose species_list locks the output-head order.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--split_mode", choices=["preserve", "random", "none"], default="preserve",
        help="Preserve and validate an existing split by default; random splitting is explicit.",
    )
    ap.add_argument("--holdout_species", nargs="+", default=[],
                    help="Species reserved for zero-shot eval; excluded from "
                         "top_species+mic_wide, written to mic_holdout.parquet.")
    args = ap.parse_args()

    integrated = Path(args.integrated)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=== 1) peptides ===")
    peptides = build_peptides(integrated)
    peptides.to_parquet(out / "peptides.parquet", index=False)

    print("\n=== 2) MIC long ===")
    valid_keys = set(peptides["dedup_key"])
    mic_long, mic_audit = build_mic_long(integrated, valid_keys=valid_keys)
    mic_long.to_parquet(out / "mic_long.parquet", index=False)

    print("\n=== 3) Top species ===")
    holdout = set(args.holdout_species or [])
    if holdout:
        overlap = holdout & set(GUARANTEED_SPECIES)
        if overlap:
            raise SystemExit(f"holdout overlaps GUARANTEED_SPECIES: {overlap}")
        print(f"holdout (excluded from top-K): {sorted(holdout)}")
    if args.top_species_file:
        panel = json.loads(Path(args.top_species_file).read_text())
        top_species = list(panel["species_list"])
        if len(top_species) != args.top_k:
            raise SystemExit(
                f"locked panel has {len(top_species)} species, expected --top_k={args.top_k}"
            )
        overlap = set(top_species) & holdout
        if overlap:
            raise SystemExit(f"locked panel overlaps holdout species: {sorted(overlap)}")
        missing = set(top_species) - set(mic_long["target_species"])
        if missing:
            raise SystemExit(f"locked panel species absent from MIC data: {sorted(missing)}")
        print(f"locked top-species order from {args.top_species_file}")
    else:
        top_species = pick_top_species(mic_long, args.top_k, exclude=holdout)
    counts = mic_long["target_species"].value_counts().to_dict()
    json.dump({"species_list": top_species,
               "n_rows_per_species": {sp: int(counts.get(sp, 0)) for sp in top_species},
               "k": args.top_k},
              open(out / "top_species.json", "w"), indent=2)
    print(f"top-{args.top_k} species ({len(top_species)}): {top_species[:10]}...")
    for sp in top_species:
        print(f"  {sp}: {counts.get(sp, 0)} rows")

    print("\n=== 4) MIC wide ===")
    mic_wide = build_mic_wide(mic_long, top_species)
    mic_wide.to_parquet(out / "mic_wide.parquet", index=False)
    print(f"mic_wide: {mic_wide.shape}")

    print("\n=== 5) Hemolysis ===")
    hemo, hemo_audit = build_hemolysis(integrated, valid_keys=valid_keys)
    hemo.to_parquet(out / "hemolysis.parquet", index=False)

    print("\n=== 6) Splits ===")
    split_path = out / "splits.parquet"
    if args.split_mode == "random":
        splits = build_splits(peptides, seed=args.seed)
        splits.to_parquet(split_path, index=False)
        print("WARNING: wrote explicitly requested random peptide split")
        print(splits["split"].value_counts().to_string())
    elif args.split_mode == "preserve":
        if not split_path.exists():
            raise SystemExit(
                "--split_mode preserve requires an existing splits.parquet; "
                "use --split_mode none and then run scripts/build_cluster_splits.py"
            )
        splits = pd.read_parquet(split_path)
        expected = set(peptides["dedup_key"])
        actual = set(splits["dedup_key"])
        if expected != actual:
            raise SystemExit(
                "existing splits.parquet keys do not match rebuilt peptides; "
                "rebuild the cluster-aware split explicitly"
            )
        print("preserved existing splits.parquet")
        print(splits["split"].value_counts().to_string())
    else:
        splits = None
        print("split generation skipped; build a cluster-aware split before training")

    if holdout:
        print("\n=== 6b) Holdout (zero-shot species eval) ===")
        assert not (holdout & set(top_species)), (
            f"bug: holdout leaked into top-K: {holdout & set(top_species)}"
        )
        ho_rows = mic_long[mic_long["target_species"].isin(holdout)].copy()
        ho_rows.to_parquet(out / "mic_holdout.parquet", index=False)
        json.dump({"species_list": list(args.holdout_species),
                   "n_rows_per_species": {sp: int((ho_rows["target_species"] == sp).sum())
                                           for sp in args.holdout_species}},
                  open(out / "holdout_species.json", "w"), indent=2)
        print(f"mic_holdout.parquet: {len(ho_rows)} rows across {len(holdout)} species")
        for sp in args.holdout_species:
            print(f"  {sp}: {int((ho_rows['target_species'] == sp).sum())} rows")

    # some summary statistics
    summary = {
        "n_peptides": int(len(peptides)),
        "n_with_mic": int(mic_wide["dedup_key"].nunique()),
        "n_with_hemo": int(len(hemo)),
        "top_species_n": len(top_species),
        "mic_long_rows": int(len(mic_long)),
        "mic_long_censored_right_frac": float(mic_long["censored_right"].mean()),
        "mic_long_censored_left_frac": float(mic_long["censored_left"].mean()),
        "mic_excluded_mixed_groups": int(mic_audit["excluded_mixed_groups"]),
        "hemolysis_excluded_mixed_groups": int(hemo_audit["excluded_mixed_groups"]),
    }
    json.dump(summary, open(out / "_summary.json", "w"), indent=2)
    json.dump(
        {"mic": mic_audit, "hemolysis": hemo_audit},
        open(out / "censoring_aggregation_audit.json", "w"),
        indent=2,
    )
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
