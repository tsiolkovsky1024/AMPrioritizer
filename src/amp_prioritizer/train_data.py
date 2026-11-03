"""Train-set construction with non-AMP negatives.

Negatives come from the AMPlify non-AMP FASTA (see scripts/fetch_negatives.py),
filtered length [5, 100], canonical AA only, deduped, then truncated to len(positives)
to give a 1:1 ratio.

For each negative we synthesize labels:
  amp_label = 0
  species_label = 0 across all top-K species  (inactive everywhere)
  mic_mask    = False for all species          (negatives do not participate in
                                                MIC regression)
  mic_censored_right = False
  mic_censored_left  = False
  mic_target  = 0.0  (placeholder, masked out)
  hemolysis_target = 0, mask=False             (heads dropped, but collator reads keys)

Why mic_mask=False: the right-censored NLL is unbounded above and its gradient pushes
mu towards +inf, so anchoring the negatives at a fixed threshold inflates predictions for
real positives that resemble them. Masking negatives out of every MIC loss avoids this
while still using them for L_amp and L_sp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO
from torch.utils.data import Sampler


from amp_prioritizer.dataset import AMPMultiTaskDataset, AMPCollator
from amp_prioritizer.schema import load_latest_species

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_amplify_negatives(
    fasta_path: str | Path,
    max_n: int,
    *,
    min_len: int = 5,
    max_len: int = 100,
) -> list[str]:
    """Load, filter and deduplicate negative sequences from a FASTA file."""
    seqs: list[str] = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        s = str(rec.seq).upper().replace("*", "")
        if min_len <= len(s) <= max_len and all(c in CANONICAL_AA for c in s):
            seqs.append(s)
    seqs = list(dict.fromkeys(seqs))
    return seqs[:max_n]


class AMPTrainWithNegatives(AMPMultiTaskDataset):
    """Train dataset = positives (split=train) + AMPlify negatives (1:1)."""

    def __init__(
        self,
        data_dir: str = "data/current",
        *,
        max_len: int = 200,
        active_threshold_log10: float = 1.5,
        physchem_cache: str | None = None,
        neg_fasta: str | None = None,
        neg_log10_mic: float = 3.91,
        exclude_dedup_keys: set[str] | None = None,
    ):
        # Build the positives dataset via the existing class.
        super().__init__(
            data_dir=data_dir,
            split="train",
            max_len=max_len,
            active_threshold_log10=active_threshold_log10,
            physchem_cache=physchem_cache,
            include_no_mic=True,
        )
        if exclude_dedup_keys:
            keep = np.asarray([
                i for i, key in enumerate(self.dedup_keys)
                if key not in exclude_dedup_keys
            ], dtype=np.int64)
            self._filter(keep)
            self.cluster_id = np.arange(len(self.seqs), dtype=np.int32)
            print(f"[AMPTrainWithNegatives] excluded={len(exclude_dedup_keys)} keys; kept={len(self.seqs)}", flush=True)
        if neg_fasta is None:
            raise ValueError("neg_fasta required")

        n_pos = len(self.seqs)
        N = self.n_species
        neg_seqs = load_amplify_negatives(neg_fasta, max_n=n_pos, min_len=5, max_len=max_len)
        n_neg = len(neg_seqs)
        print(f"[AMPTrainWithNegatives] positives={n_pos}  negatives={n_neg}  "
              f"(negatives masked out of MIC losses; participate in L_amp + L_sp only)",
              flush=True)

        # Append synthetic negative rows matching the parent's per-row arrays.
        self.seqs = list(self.seqs) + list(neg_seqs)
        self.dedup_keys = list(self.dedup_keys) + [f"_neg_{i}" for i in range(n_neg)]
        self.amp_label = np.concatenate([
            self.amp_label,
            np.zeros(n_neg, dtype=self.amp_label.dtype),
        ])
        # Species: 0 = inactive everywhere (NOT -1 — we want the BCE to actually fire).
        self.species_label = np.concatenate([
            self.species_label,
            np.zeros((n_neg, N), dtype=self.species_label.dtype),
        ], axis=0)
        # MIC: mask=False for all negatives — they do NOT participate in any MIC loss.
        # The neg_log10_mic value is kept as a placeholder for backwards-compat but ignored.
        _ = neg_log10_mic  # parameter retained for config compatibility; value unused
        self.mic = np.concatenate([
            self.mic,
            np.zeros((n_neg, N), dtype=self.mic.dtype),
        ], axis=0)
        self.mic_mask = np.concatenate([
            self.mic_mask, np.zeros((n_neg, N), dtype=self.mic_mask.dtype),
        ], axis=0)
        self.mic_cr = np.concatenate([
            self.mic_cr, np.zeros((n_neg, N), dtype=self.mic_cr.dtype),
        ], axis=0)
        self.mic_cl = np.concatenate([
            self.mic_cl, np.zeros((n_neg, N), dtype=self.mic_cl.dtype),
        ], axis=0)
        # Hemolysis fields: zero / mask=False (heads dropped, collator still reads).
        self.hemo = np.concatenate([self.hemo, np.zeros(n_neg, dtype=self.hemo.dtype)])
        self.hemo_mask = np.concatenate([
            self.hemo_mask, np.zeros(n_neg, dtype=self.hemo_mask.dtype),
        ])
        self.hemo_cr = np.concatenate([
            self.hemo_cr, np.zeros(n_neg, dtype=self.hemo_cr.dtype),
        ])
        self.hemo_cl = np.concatenate([
            self.hemo_cl, np.zeros(n_neg, dtype=self.hemo_cl.dtype),
        ])
        # Physchem: compute on negatives via the same path the parent used (no cache).
        from amp_prioritizer.physchem import compute_many
        neg_physchem = compute_many(neg_seqs, cache_path=None).astype(np.float32)
        self.physchem = np.concatenate([self.physchem, neg_physchem], axis=0)
        self.cluster_id = np.arange(len(self.seqs), dtype=np.int32)


class SpeciesGroupedBatchSampler(Sampler[list[int]]):
    """Build batches with usable within-species MIC ranking pairs.

    The grouped portion is sampled uniformly over eligible species. Remaining
    slots preserve random examples and synthetic negatives for the AMP head.
    """

    def __init__(
        self,
        dataset: AMPMultiTaskDataset,
        batch_size: int,
        *,
        species_per_batch: int = 2,
        samples_per_species: int = 2,
        negative_fraction: float = 0.5,
        seed: int = 42,
        drop_last: bool = True,
    ):
        if batch_size < species_per_batch * samples_per_species:
            raise ValueError("batch_size is smaller than the grouped MIC allocation")
        if not 0.0 <= negative_fraction <= 1.0:
            raise ValueError("negative_fraction must be in [0, 1]")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.species_per_batch = int(species_per_batch)
        self.samples_per_species = int(samples_per_species)
        self.negative_fraction = float(negative_fraction)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        exact = dataset.mic_mask & ~dataset.mic_cr & ~dataset.mic_cl
        self.species_indices = [np.flatnonzero(exact[:, i]) for i in range(exact.shape[1])]
        self.eligible_species = np.asarray([
            i for i, idx in enumerate(self.species_indices)
            if len(idx) >= self.samples_per_species
        ], dtype=np.int64)
        if len(self.eligible_species) < self.species_per_batch:
            raise ValueError("not enough species with exact MIC rows for grouped batches")
        self.all_indices = np.arange(len(dataset), dtype=np.int64)
        self.negative_indices = np.flatnonzero(~dataset.mic_mask.any(axis=1))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(len(self)):
            species = rng.choice(
                self.eligible_species, size=self.species_per_batch, replace=False,
            )
            batch: list[int] = []
            for sp in species:
                pool = self.species_indices[int(sp)]
                chosen = rng.choice(
                    pool, size=self.samples_per_species,
                    replace=len(pool) < self.samples_per_species,
                )
                batch.extend(int(x) for x in chosen)

            remaining = self.batch_size - len(batch)
            n_neg = min(
                len(self.negative_indices),
                int(round(remaining * self.negative_fraction)),
            )
            if n_neg:
                neg = rng.choice(
                    self.negative_indices, size=n_neg,
                    replace=len(self.negative_indices) < n_neg,
                )
                batch.extend(int(x) for x in neg)
            n_random = self.batch_size - len(batch)
            if n_random:
                extra = rng.choice(
                    self.all_indices, size=n_random,
                    replace=len(self.all_indices) < n_random,
                )
                batch.extend(int(x) for x in extra)
            rng.shuffle(batch)
            yield batch
        self.epoch += 1


def species_frequency_weights(dataset: AMPMultiTaskDataset) -> np.ndarray:
    """Return inverse-sqrt exact-MIC weights, normalized to mean one."""
    exact = dataset.mic_mask & ~dataset.mic_cr & ~dataset.mic_cl
    counts = exact.sum(axis=0).astype(np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    valid = counts > 0
    weights[valid] = 1.0 / np.sqrt(counts[valid])
    if valid.any():
        weights[valid] /= weights[valid].mean()
    return weights.astype(np.float32)


def make_train_val_test(
    data_dir: str,
    *,
    max_len: int,
    physchem_cache_dir: str,
    active_threshold_log10: float,
    neg_fasta: str,
    neg_log10_mic: float,
    exclude_dedup_keys: set[str] | None = None,
):
    """Train (positives + AMPlify negatives), val (positives only), test (positives only)."""
    train = AMPTrainWithNegatives(
        data_dir=data_dir,
        max_len=max_len,
        active_threshold_log10=active_threshold_log10,
        physchem_cache=str(Path(physchem_cache_dir) / "train.npz"),
        neg_fasta=neg_fasta,
        neg_log10_mic=neg_log10_mic,
        exclude_dedup_keys=exclude_dedup_keys,
    )
    val = AMPMultiTaskDataset(
        data_dir=data_dir, split="val", max_len=max_len,
        active_threshold_log10=active_threshold_log10,
        physchem_cache=str(Path(physchem_cache_dir) / "val.npz"),
    )
    test = AMPMultiTaskDataset(
        data_dir=data_dir, split="test", max_len=max_len,
        active_threshold_log10=active_threshold_log10,
        physchem_cache=str(Path(physchem_cache_dir) / "test.npz"),
    )
    return {"train": train, "val": val, "test": test}


__all__ = [
    "AMPTrainWithNegatives", "AMPCollator", "SpeciesGroupedBatchSampler",
    "make_train_val_test", "load_amplify_negatives", "load_latest_species",
    "species_frequency_weights",
]
