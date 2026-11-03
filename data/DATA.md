# Dataset: `v2_censor_safe`

This directory holds the processed peptide-identity, MIC and split tables used by every
experiment in the paper. Protocol name: `canonical_mmseqs80_v2_censor_safe`.

All files are listed in [`manifest.json`](manifest.json) with byte size and SHA-256.
Verify a checkout with:

```bash
amp-prioritizer data verify
# or: python -m amp_prioritizer.data_versions verify
```

## Files

| File | Rows | Description |
|---|---|---|
| `current/peptides.parquet` | 68,421 | Deduplicated, sanitized peptide sequences. Columns: `dedup_key`, `seq`, `length`, `source_list`, `has_modification`, `n_sources`. |
| `current/mic_long.parquet` | 106,728 | One row per peptide–species observation unit. Columns: `dedup_key`, `target_species`, `target_group`, `mic_uM_geomean`, `log10_mic`, `censored_right`, `censored_left`, `observation_type`, `aggregation_method`, `n_measurements`, `n_exact`, `n_right`, `n_left`. |
| `current/mic_wide.parquet` | 20,126 | Wide view over the 20-species panel: `mic_uM/`, `log10_mic/`, `censored_right/`, `censored_left/`, `n_measurements/` per species. |
| `current/splits.parquet` | 68,421 | `dedup_key` → `split` (`train` / `validation` / `test`). |
| `current/cluster_assignments.parquet` | 68,421 | MMseqs2 80 % identity / 80 % coverage assignment: `dedup_key`, `seq`, `cluster_rep`, `split`. |
| `current/clusters_mmseqs80.tsv` | 30,782 | Raw cluster table (cluster representative → members). |
| `current/top_species.json` | 20 | The censoring-evaluation species panel, ordered by MIC record count. |
| `current/holdout_species_v2.json` | 5 | Target organisms for the target-conditioned ranking study. |
| `current/mic_holdout_v2.parquet` | 414 | Long-format MIC rows for the five ranking-study organisms. |
| `current/species_kmer_features.npz` | 20 × 340 | Genome 4-mer frequency features for the 20-species panel. |
| `current/species_kmer_features_with_holdout.npz` | 25 × 340 | Same features extended to the five ranking-study organisms. |
| `current/physchem/{train,val,test}.npz` | — | Cached physicochemical descriptors. |
| `current/external_binary/` | — | Binary AMP/non-AMP benchmark and the *E. coli* / *S. aureus* MIC tasks (`seq`, `label` / `log10_mic`, `pmic`). |
| `current/censoring_aggregation_audit.json` | — | Replicate-aggregation accounting (mixed-report groups excluded, sanitization failures). |

## Counts

| Quantity | Value |
|---|---|
| Unique sanitized sequences | 68,421 |
| Peptide–species observation units | 106,728 |
| Exact | 83,516 (78.25 %) |
| Right-censored | 22,742 (21.31 %) |
| Left-censored | 470 (0.44 %) |
| Sequence clusters (MMseqs2, 80 % id / 80 % cov) | 30,782 |
| Cluster cross-split leaks | 0 |

Split sizes — peptides 55,269 / 6,553 / 6,599; MIC-labelled peptides 17,546 / 2,074 / 1,946;
peptide–species units 86,799 / 10,796 / 9,133 (train / validation / test).

## Aggregation rule

Repeated exact measurements are combined by geometric mean. Repeated right-censored
measurements keep the maximum lower bound; repeated left-censored measurements keep the
minimum upper bound. Groups that mix observation types are excluded, because the row
schema cannot represent their joint likelihood without information loss (3,834 groups).

## Provenance

Sequence identity records were integrated from APD, DRAMP, dbAMP, DBAASP and CAMPR4.
MIC measurements derive from DBAASP (117,248 source rows), CAMPR4 (61,301) and DRAMP
(13,498). Reference genomes were downloaded from NCBI RefSeq using the accessions
recorded in `scripts/download_genomes.py`.

Redistribution terms differ between the underlying resources. DRAMP is released under
CC BY 4.0; APD, dbAMP, DBAASP and CAMPR4 each state their own terms. This release
distributes the processed tables for reproducibility of the published analysis and
attributes every source above; it makes no claim about the redistribution terms of the
upstream databases, and users who need the raw records should obtain them from the
original resources.

## Non-AMP negatives

The binary and benchmark training sets use the non-AMP FASTA distributed with AMPlify
(GPL-3.0), which is not redistributed here. Fetch it with:

```bash
python scripts/fetch_negatives.py            # writes data/negatives/
```

The script checks the download against the SHA-256 of the file used in the paper
(`b8d5e546…0269f921`) and refuses to continue on a mismatch.

## Optional Stage-0 corpus

`configs/stage0_mlm.yaml` points at `data/mlm_corpus/peptideatlas_cdhit90.fasta`.
That corpus is optional: when the file is absent, Stage 0 adapts on the AMP sequences
in `peptides.parquet` alone.
