"""Shared species feature encoders and peptide-species fusion layers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LEGACY_KMER_DIM = 340


def load_species_feature_tensor(
    path: str | Path,
    species_list: list[str] | None,
    *,
    expected_dim: int | None = None,
) -> torch.Tensor:
    if species_list is None:
        raise ValueError("species_list is required for a feature-backed species encoder")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    data = np.load(source, allow_pickle=True)
    if not {"names", "features"} <= set(data.files):
        raise ValueError(f"{source}: expected names and features arrays")
    names = [str(name) for name in data["names"]]
    features = np.asarray(data["features"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"{source}: features must be 2-D, got {features.shape}")
    if expected_dim is not None and features.shape[1] != expected_dim:
        raise ValueError(
            f"{source}: expected feature width {expected_dim}, got {features.shape[1]}"
        )
    name_to_idx = {name: index for index, name in enumerate(names)}
    missing = [species for species in species_list if species not in name_to_idx]
    if missing:
        raise KeyError(f"{source}: missing species {missing[:5]}")
    rows = np.stack([features[name_to_idx[species]] for species in species_list])
    return torch.from_numpy(rows)


class SpeciesFeatureEncoder(nn.Module):
    """Project a fixed per-species feature table into the PLM hidden space."""

    def __init__(
        self,
        features: torch.Tensor,
        hidden: int,
        mlp_hidden: int = 512,
        dropout: float = 0.0,
        layout: str = "normalized",
    ) -> None:
        super().__init__()
        if features.ndim != 2:
            raise ValueError(f"expected [species, features], got {tuple(features.shape)}")
        self.register_buffer("features", features.float().contiguous())
        self.layout = layout
        if layout == "legacy":
            self.mlp = nn.Sequential(
                nn.Linear(int(features.shape[1]), mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, hidden),
            )
        elif layout == "normalized":
            self.mlp = nn.Sequential(
                nn.LayerNorm(int(features.shape[1])),
                nn.Linear(int(features.shape[1]), mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, hidden),
            )
        else:
            raise ValueError(f"unknown species encoder layout: {layout}")
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    @property
    def n_species(self) -> int:
        return int(self.features.shape[0])

    def forward(self) -> torch.Tensor:
        return self.mlp(self.features)

    @torch.no_grad()
    def extend(self, extra: torch.Tensor) -> None:
        if extra.ndim != 2 or extra.shape[1] != self.features.shape[1]:
            raise ValueError(
                f"extra feature shape {tuple(extra.shape)} is incompatible with "
                f"{tuple(self.features.shape)}"
            )
        self.features = torch.cat(
            [self.features, extra.to(self.features.device, self.features.dtype)], dim=0
        ).contiguous()


class SpeciesFusion(nn.Module):
    """Produce a stable 2H peptide-species representation for all fusion modes."""

    MODES = {"concat", "film", "gated_residual"}

    def __init__(self, hidden: int, mode: str = "concat") -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"species_fusion must be one of {sorted(self.MODES)}, got {mode!r}")
        self.mode = mode
        self.norm = nn.LayerNorm(hidden) if mode != "concat" else nn.Identity()
        self.scale = nn.Linear(hidden, hidden) if mode != "concat" else None
        self.shift = nn.Linear(hidden, hidden) if mode != "concat" else None
        self.gate = nn.Linear(hidden, hidden) if mode == "gated_residual" else None
        if self.scale is not None:
            nn.init.zeros_(self.scale.weight)
            nn.init.zeros_(self.scale.bias)
            nn.init.zeros_(self.shift.weight)
            nn.init.zeros_(self.shift.bias)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, peptide: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        batch, hidden = peptide.shape
        n_species = species.shape[0]
        peptide_rows = peptide.unsqueeze(1).expand(batch, n_species, hidden)
        species_rows = species.unsqueeze(0).expand(batch, n_species, hidden)
        if self.mode == "concat":
            conditioned = peptide_rows
        else:
            normalized = self.norm(peptide_rows)
            delta = torch.tanh(self.scale(species_rows)) * normalized + self.shift(species_rows)
            if self.mode == "film":
                conditioned = peptide_rows + delta
            else:
                conditioned = peptide_rows + torch.sigmoid(self.gate(species_rows)) * delta
        return torch.cat([conditioned, species_rows], dim=-1)

    def paired(self, peptide: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        """Fuse row-aligned peptide/species embeddings instead of their Cartesian product."""
        if peptide.shape != species.shape or peptide.ndim != 2:
            raise ValueError(
                f"paired fusion requires matching [B,H] tensors, got "
                f"{tuple(peptide.shape)} and {tuple(species.shape)}"
            )
        if self.mode == "concat":
            conditioned = peptide
        else:
            normalized = self.norm(peptide)
            delta = torch.tanh(self.scale(species)) * normalized + self.shift(species)
            if self.mode == "film":
                conditioned = peptide + delta
            else:
                conditioned = peptide + torch.sigmoid(self.gate(species)) * delta
        return torch.cat([conditioned, species], dim=-1)
