"""v3 Dataset / Collator — reads data/current/ produced by amp_prioritizer/ingest.py.

Inputs (under --data_dir, default data/current/):
  peptides.parquet         68k unique peptides (dedup_key, seq, length, source_list, ...)
  mic_wide.parquet         20k peptides × top-K species (mic/log10_mic/censored_{right,left}/n)
  hemolysis.parquet        9k peptides with HC50
  splits.parquet           random 80/10/10 by dedup_key
  top_species.json         dynamic top-K species list (K=20)
  physchem/                cached physchem .npz per split

Every __getitem__ returns:
  seq                       str
  amp_label                 int          (all 1 in v3 — no non-AMP pool)
  species_label             int64[N]     1 = active (log10_mic ≤ thr), 0 = inactive (censored_right), -1 = unknown
  mic_target                float32[N]   log10(µM)
  mic_mask                  bool[N]
  mic_censored_right        bool[N]
  mic_censored_left         bool[N]
  hemolysis_target          float32      log10(µM) HC50
  hemolysis_mask            bool
  hemolysis_censored_right  bool
  hemolysis_censored_left   bool
  physchem_target           float32[5]
  cluster_id                int          v3 has no MMseqs2 cluster — degenerates to per-row id

Note — legacy fields removed in v3:
  * source_filter (v3 ingests all sources uniformly)
  * conflict_group_id sampler support (ClusterAwareBatchSampler is gone)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


from amp_prioritizer.schema import PHYSCHEM_FIELDS, load_latest_species
from amp_prioritizer.physchem import compute_many

N_PHYSCHEM = len(PHYSCHEM_FIELDS)


def _active_label_from_mic(log10_mic: float, censored_right: bool,
                           active_threshold: float = 1.5) -> int:
    """log10_mic ≤ 1.5 (MIC ≤ 32 µM) → active=1; right-censored → inactive=0; else -1."""
    if np.isnan(log10_mic):
        return -1
    if censored_right:
        return 0
    if log10_mic <= active_threshold:
        return 1
    return 0


class AMPMultiTaskDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path = "data/current",
        split: str | None = None,                  # train | val | test | None
        max_len: int = 200,
        active_threshold_log10: float = 1.5,
        physchem_cache: str | Path | None = None,
        include_no_mic: bool = True,
    ):
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.species_list = load_latest_species(str(data_dir))
        self.n_species = len(self.species_list)
        self.max_len = max_len
        self.active_threshold_log10 = active_threshold_log10

        peptides = pd.read_parquet(data_dir / "peptides.parquet")
        splits = pd.read_parquet(data_dir / "splits.parquet")
        peptides = peptides.merge(splits, on="dedup_key", how="left")
        if split is not None:
            peptides = peptides[peptides["split"] == split].reset_index(drop=True)
        peptides = peptides[(peptides["length"] >= 3) & (peptides["length"] <= max_len)].reset_index(drop=True)

        mic_wide = pd.read_parquet(data_dir / "mic_wide.parquet").set_index("dedup_key")
        hemo = pd.read_parquet(data_dir / "hemolysis.parquet")
        if len(hemo) > 0:
            hemo = hemo.set_index("dedup_key")

        n = len(peptides)
        self.seqs: list[str] = peptides["seq"].tolist()
        self.dedup_keys: list[str] = peptides["dedup_key"].tolist()
        self.amp_label = np.ones(n, dtype=np.int8)

        self.species_label = np.full((n, self.n_species), -1, dtype=np.int8)
        self.mic = np.zeros((n, self.n_species), dtype=np.float32)
        self.mic_mask = np.zeros((n, self.n_species), dtype=np.bool_)
        self.mic_cr = np.zeros((n, self.n_species), dtype=np.bool_)
        self.mic_cl = np.zeros((n, self.n_species), dtype=np.bool_)

        for i, dk in enumerate(self.dedup_keys):
            if dk not in mic_wide.index:
                continue
            row = mic_wide.loc[dk]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for j, sp in enumerate(self.species_list):
                log10_mic = row.get(f"log10_mic/{sp}")
                if pd.isna(log10_mic):
                    continue
                self.mic_mask[i, j] = True
                self.mic[i, j] = float(log10_mic)
                cr = bool(row.get(f"censored_right/{sp}", False))
                cl = bool(row.get(f"censored_left/{sp}", False))
                self.mic_cr[i, j] = cr
                self.mic_cl[i, j] = cl
                self.species_label[i, j] = _active_label_from_mic(
                    float(log10_mic), cr, active_threshold=active_threshold_log10
                )

        self.hemo = np.zeros(n, dtype=np.float32)
        self.hemo_mask = np.zeros(n, dtype=np.bool_)
        self.hemo_cr = np.zeros(n, dtype=np.bool_)
        self.hemo_cl = np.zeros(n, dtype=np.bool_)
        if len(hemo) > 0:
            for i, dk in enumerate(self.dedup_keys):
                if dk not in hemo.index:
                    continue
                r = hemo.loc[dk]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                self.hemo[i] = float(r["log10_hc50"])
                self.hemo_mask[i] = True
                self.hemo_cr[i] = bool(r.get("censored_right", False))
                self.hemo_cl[i] = bool(r.get("censored_left", False))

        if not include_no_mic:
            keep = self.mic_mask.any(axis=1) | self.hemo_mask
            self._filter(np.where(keep)[0])

        self.physchem = compute_many(self.seqs, cache_path=physchem_cache).astype(np.float32)
        self.cluster_id = np.arange(len(self.seqs), dtype=np.int32)

    def _filter(self, idx: np.ndarray) -> None:
        self.seqs = [self.seqs[i] for i in idx]
        self.dedup_keys = [self.dedup_keys[i] for i in idx]
        self.amp_label = self.amp_label[idx]
        self.species_label = self.species_label[idx]
        self.mic = self.mic[idx]
        self.mic_mask = self.mic_mask[idx]
        self.mic_cr = self.mic_cr[idx]
        self.mic_cl = self.mic_cl[idx]
        self.hemo = self.hemo[idx]
        self.hemo_mask = self.hemo_mask[idx]
        self.hemo_cr = self.hemo_cr[idx]
        self.hemo_cl = self.hemo_cl[idx]
        if hasattr(self, "physchem"):
            self.physchem = self.physchem[idx]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict:
        return {
            "seq": self.seqs[idx][: self.max_len],
            "amp_label": int(self.amp_label[idx]),
            "species_label": self.species_label[idx].astype(np.int64),
            "mic_target": self.mic[idx].astype(np.float32),
            "mic_mask": self.mic_mask[idx].astype(np.bool_),
            "mic_censored_right": self.mic_cr[idx].astype(np.bool_),
            "mic_censored_left": self.mic_cl[idx].astype(np.bool_),
            "hemolysis_target": float(self.hemo[idx]),
            "hemolysis_mask": bool(self.hemo_mask[idx]),
            "hemolysis_censored_right": bool(self.hemo_cr[idx]),
            "hemolysis_censored_left": bool(self.hemo_cl[idx]),
            "physchem_target": self.physchem[idx].astype(np.float32),
            "cluster_id": int(self.cluster_id[idx]),
        }


class AMPCollator:
    """Tokenize a batch and stack all per-row tensors."""

    def __init__(self, tokenizer, max_len: int = 202):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch: Sequence[dict]) -> dict:
        seqs = [b["seq"] for b in batch]
        enc = self.tok(seqs, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "amp_label": torch.tensor([b["amp_label"] for b in batch], dtype=torch.long),
            "species_label": torch.tensor(np.stack([b["species_label"] for b in batch]),
                                          dtype=torch.long),
            "mic_target": torch.tensor(np.stack([b["mic_target"] for b in batch]),
                                       dtype=torch.float32),
            "mic_mask": torch.tensor(np.stack([b["mic_mask"] for b in batch]),
                                     dtype=torch.bool),
            "mic_censored_right": torch.tensor(np.stack([b["mic_censored_right"] for b in batch]),
                                               dtype=torch.bool),
            "mic_censored_left": torch.tensor(np.stack([b["mic_censored_left"] for b in batch]),
                                              dtype=torch.bool),
            "hemolysis_target": torch.tensor([b["hemolysis_target"] for b in batch],
                                             dtype=torch.float32),
            "hemolysis_mask": torch.tensor([b["hemolysis_mask"] for b in batch], dtype=torch.bool),
            "hemolysis_censored_right": torch.tensor([b["hemolysis_censored_right"] for b in batch],
                                                     dtype=torch.bool),
            "hemolysis_censored_left": torch.tensor([b["hemolysis_censored_left"] for b in batch],
                                                    dtype=torch.bool),
            "physchem_target": torch.tensor(np.stack([b["physchem_target"] for b in batch]),
                                            dtype=torch.float32),
            "cluster_id": torch.tensor([b["cluster_id"] for b in batch], dtype=torch.long),
        }
        return out


def make_datasets(data_dir: str = "data/current", max_len: int = 200,
                  physchem_cache_dir: str = "data/current/physchem",
                  active_threshold_log10: float = 1.5,
                  ) -> dict[str, AMPMultiTaskDataset]:
    Path(physchem_cache_dir).mkdir(parents=True, exist_ok=True)
    return {
        "train": AMPMultiTaskDataset(
            data_dir, split="train", max_len=max_len,
            physchem_cache=Path(physchem_cache_dir) / "train.npz",
            active_threshold_log10=active_threshold_log10),
        "val": AMPMultiTaskDataset(
            data_dir, split="val", max_len=max_len,
            physchem_cache=Path(physchem_cache_dir) / "val.npz",
            active_threshold_log10=active_threshold_log10),
        "test": AMPMultiTaskDataset(
            data_dir, split="test", max_len=max_len,
            physchem_cache=Path(physchem_cache_dir) / "test.npz",
            active_threshold_log10=active_threshold_log10),
    }
