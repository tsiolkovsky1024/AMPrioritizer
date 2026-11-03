"""ESM-2 + optional LoRA + multi-head AMP model (v3).

Heads:
  amp_head          pooled → logit
  species_head      [pooled || species_rep] → logit per species
  species_mic_head  [pooled || species_rep] → (μ, log σ) per species  (Tobit/censored NLL)
  physchem_head     pooled → 5-dim closed-form target
  hemolysis_head    pooled → (μ, log σ)                               (Tobit/censored NLL)

The source of species_rep is determined by `species_encoder_cfg["type"]`:
  embedding  — nn.Embedding(N, hidden), the default fallback
  kmer_mlp   — genomic k-mer features (340d) → MLP → hidden, following LLAMP
               requires `kmer_features_path` pointing at the npz from amp_prioritizer.species_genome

v3 drops the v1/v2 "embedding + 3 heads" baseline. The MIC head is always 2-dim.
Species count is dynamic; pass `n_species` from `load_latest_species()`.

Fine-tune mode:
  finetune_mode="full"  (default, recommended for 8×V100 32GB) — backbone fully trainable
  finetune_mode="lora"  — LoRA r=16 on q/k/v/dense, backbone frozen
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer


from amp_prioritizer.schema import PHYSCHEM_FIELDS
from amp_prioritizer.species_conditioning import (
    LEGACY_KMER_DIM,
    SpeciesFeatureEncoder,
    SpeciesFusion,
    load_species_feature_tensor,
)

N_PHYSCHEM = len(PHYSCHEM_FIELDS)
KMER_DIM = LEGACY_KMER_DIM


class MaskedMeanPool(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        s = (hidden * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return s / denom


class MLPHead(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KMerSpeciesEncoder(nn.Module):
    """LLAMP-style species encoder: 340-d genomic k-mer features → MLP → hidden.

    Features are stored as a buffer (non-learnable); the projection MLP is learnable.
    `extend(extra)` appends new species rows at eval time (e.g. for holdout eval).
    """

    def __init__(self, kmer_features: torch.Tensor, hidden: int, mlp_hidden: int = 512):
        super().__init__()
        assert kmer_features.ndim == 2 and kmer_features.shape[1] == KMER_DIM, (
            f"expected [N, {KMER_DIM}], got {tuple(kmer_features.shape)}"
        )
        self.register_buffer("features", kmer_features.float().contiguous())
        self.mlp = nn.Sequential(
            nn.Linear(KMER_DIM, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden),
        )
        # Small init: keeps the species condition from overpowering the ESM-2
        # representation in early training
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    @property
    def n_species(self) -> int:
        return int(self.features.shape[0])

    def forward(self) -> torch.Tensor:
        return self.mlp(self.features)  # [N, hidden]

    @torch.no_grad()
    def extend(self, extra: torch.Tensor) -> None:
        """Append new species rows to the feature buffer (eval-only)."""
        assert extra.ndim == 2 and extra.shape[1] == KMER_DIM, extra.shape
        extra = extra.to(self.features.device, dtype=self.features.dtype)
        self.features = torch.cat([self.features, extra], dim=0).contiguous()


def _normalize_species_encoder_cfg(cfg: dict | None) -> dict:
    cfg = dict(cfg) if cfg else {}
    cfg.setdefault("type", "embedding")
    return cfg


def _load_kmer_features_as_tensor(
    path: str, species_list: list[str] | None
) -> torch.Tensor:
    if species_list is None:
        raise ValueError("species_list is required when type=kmer_mlp")
    data = np.load(path, allow_pickle=True)
    names: list[str] = list(data["names"])
    features: np.ndarray = data["features"]
    if features.shape[1] != KMER_DIM:
        raise ValueError(
            f"{path}: expected features width {KMER_DIM}, got {features.shape[1]}"
        )
    name_to_idx = {n: i for i, n in enumerate(names)}
    missing = [s for s in species_list if s not in name_to_idx]
    if missing:
        raise KeyError(
            f"{path} is missing species: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    rows = np.stack([features[name_to_idx[s]] for s in species_list])
    return torch.from_numpy(rows.astype(np.float32))


class AMPModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        *,
        n_species: int,
        finetune_mode: str = "full",             # "full" | "lora"
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        head_hidden: int = 512,
        head_dropout: float = 0.2,
        gradient_checkpointing: bool = True,
        hf_cache_dir: str | None = None,
        species_encoder_cfg: dict | None = None,
        species_list: list[str] | None = None,
        species_fusion: str = "concat",
        mic_head_mode: str = "distribution",
    ):
        super().__init__()
        kwargs = {"cache_dir": hf_cache_dir} if hf_cache_dir else {}
        self.backbone = AutoModel.from_pretrained(backbone_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, **kwargs)
        hidden = self.backbone.config.hidden_size

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False

        self.finetune_mode = finetune_mode
        if finetune_mode == "lora":
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type="FEATURE_EXTRACTION",
                target_modules=["query", "key", "value", "dense"],
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)
            if gradient_checkpointing and hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()
        elif finetune_mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            if gradient_checkpointing and hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()
        else:
            raise ValueError(f"finetune_mode must be 'full' or 'lora', got {finetune_mode!r}")

        self.n_species = int(n_species)
        self.pool = MaskedMeanPool()
        self.amp_head = MLPHead(hidden, head_hidden, 1, dropout=head_dropout)

        enc_cfg = _normalize_species_encoder_cfg(species_encoder_cfg)
        self.species_encoder_type = enc_cfg["type"]
        if self.species_encoder_type == "kmer_mlp":
            path = enc_cfg.get("kmer_features_path")
            if not path or not os.path.exists(path):
                print(f"[AMPModel] WARNING: kmer_features_path missing "
                      f"({path!r}); falling back to embedding encoder")
                self.species_encoder_type = "embedding"
                self.species_encoder = None
                self.species_emb = nn.Embedding(self.n_species, hidden)
            else:
                feats = load_species_feature_tensor(path, species_list, expected_dim=KMER_DIM)
                if feats.shape[0] != self.n_species:
                    raise ValueError(
                        f"kmer features cover {feats.shape[0]} species but "
                        f"n_species={self.n_species} (check species_list order)"
                    )
                self.species_encoder = SpeciesFeatureEncoder(
                    feats, hidden,
                    mlp_hidden=enc_cfg.get("mlp_hidden", 512),
                    dropout=enc_cfg.get("dropout", 0.0),
                    layout=enc_cfg.get("layout", "normalized"),
                )
                self.species_emb = None
        elif self.species_encoder_type == "embedding":
            self.species_encoder = None
            self.species_emb = nn.Embedding(self.n_species, hidden)
        else:
            raise ValueError(
                f"species_encoder.type must be embedding or kmer_mlp, "
                f"got {self.species_encoder_type!r}"
            )

        self.species_fusion_mode = species_fusion
        self.mic_head_mode = mic_head_mode
        if mic_head_mode not in {"distribution", "dual_residual"}:
            raise ValueError("mic_head_mode must be 'distribution' or 'dual_residual'")
        self.species_fusion = SpeciesFusion(hidden, species_fusion)
        self.species_head = MLPHead(hidden * 2, head_hidden, 1, dropout=head_dropout)
        self.species_mic_head = MLPHead(hidden * 2, head_hidden, 2, dropout=head_dropout)
        if mic_head_mode == "dual_residual":
            self.mic_global_head = MLPHead(hidden, head_hidden, 1, dropout=head_dropout)
            self.mic_residual_head = MLPHead(hidden * 2, head_hidden, 1, dropout=head_dropout)
        else:
            self.mic_global_head = None
            self.mic_residual_head = None
        self.physchem_head = MLPHead(hidden, head_hidden, N_PHYSCHEM, dropout=head_dropout)
        self.hemolysis_head = MLPHead(hidden, head_hidden, 2, dropout=head_dropout)

    @property
    def hidden_size(self) -> int:
        bb = self.backbone
        if hasattr(bb, "config"):
            return bb.config.hidden_size
        return bb.base_model.model.config.hidden_size

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        pooled = self.encode(input_ids, attention_mask)
        amp_logit = self.amp_head(pooled).squeeze(-1)

        if self.species_encoder is not None:
            sp_emb = self.species_encoder()                        # [N, hidden]
            N = sp_emb.shape[0]
        else:
            sp_idx = torch.arange(self.n_species, device=pooled.device)
            sp_emb = self.species_emb(sp_idx)
            N = self.n_species
        feat = self.species_fusion(pooled, sp_emb)

        species_logit = self.species_head(feat).squeeze(-1)
        mic_out = self.species_mic_head(feat)
        mic_mu = mic_out[..., 0]
        mic_log_sigma = mic_out[..., 1]
        mic_point = mic_mu
        if self.mic_head_mode == "dual_residual":
            mic_point = (
                self.mic_global_head(pooled).expand(-1, N)
                + self.mic_residual_head(feat).squeeze(-1)
            )

        physchem_pred = self.physchem_head(pooled)
        hemo_out = self.hemolysis_head(pooled)

        return {
            "amp_logit": amp_logit,
            "species_logit": species_logit,
            "mic_mu": mic_mu,
            "mic_log_sigma": mic_log_sigma,
            "mic_point": mic_point,
            "physchem_pred": physchem_pred,
            "hemo_mu": hemo_out[..., 0],
            "hemo_log_sigma": hemo_out[..., 1],
            "pooled": pooled,
        }

    def trainable_param_count(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    # ------------------------------------------------------------------
    @staticmethod
    def from_checkpoint(
        ckpt_path: str,
        backbone_name: str,
        *,
        n_species: int,
        finetune_mode: str = "full",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        head_hidden: int = 512,
        head_dropout: float = 0.2,
        gradient_checkpointing: bool = True,
        hf_cache_dir: str | None = None,
        species_encoder_cfg: dict | None = None,
        species_list: list[str] | None = None,
        species_fusion: str = "concat",
        mic_head_mode: str = "distribution",
    ) -> "AMPModel":
        """Warm-start from a v3 ckpt. If n_species in the ckpt mismatches, species heads
        are re-initialized (a WARNING is printed). Swapping species_encoder type
        (embedding ↔ kmer_mlp) also triggers re-init via the same shape-skip path."""
        state = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else None
        if state is not None:
            species_fusion = state.get("species_fusion", species_fusion)
            mic_head_mode = state.get("mic_head_mode", mic_head_mode)
            checkpoint_state = state.get("model", state)
            encoder_cfg = dict(species_encoder_cfg or {})
            if encoder_cfg.get("type") == "kmer_mlp" and "layout" not in encoder_cfg:
                first_weight = checkpoint_state.get("species_encoder.mlp.0.weight")
                if first_weight is not None:
                    encoder_cfg["layout"] = "legacy" if first_weight.ndim == 2 else "normalized"
                    print(f"[from_checkpoint] inferred species_encoder.layout={encoder_cfg['layout']}")
            species_encoder_cfg = encoder_cfg
        model = AMPModel(
            backbone_name=backbone_name,
            n_species=n_species,
            finetune_mode=finetune_mode,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            head_hidden=head_hidden, head_dropout=head_dropout,
            gradient_checkpointing=gradient_checkpointing,
            hf_cache_dir=hf_cache_dir,
            species_encoder_cfg=species_encoder_cfg,
            species_list=species_list,
            species_fusion=species_fusion,
            mic_head_mode=mic_head_mode,
        )
        if not os.path.exists(ckpt_path):
            print(f"[from_checkpoint] ckpt {ckpt_path} missing; skipping warm-start")
            return model

        assert state is not None
        sd_old = state.get("model", state)
        new_sd = model.state_dict()

        for key in list(sd_old.keys()):
            if key in new_sd and sd_old[key].shape != new_sd[key].shape:
                print(f"[from_checkpoint] shape mismatch {key}: "
                      f"ckpt={tuple(sd_old[key].shape)} new={tuple(new_sd[key].shape)} — skipping")
                sd_old.pop(key)

        missing, unexpected = model.load_state_dict(sd_old, strict=False)
        if missing:
            print(f"[from_checkpoint] {len(missing)} missing keys (defaults kept)")
            for k in missing[:5]:
                print(f"    {k}")
        if unexpected:
            print(f"[from_checkpoint] {len(unexpected)} unexpected keys (dropped)")
            for k in unexpected[:5]:
                print(f"    {k}")
        model.checkpoint_load_report = {
            "missing": list(missing), "unexpected": list(unexpected)}
        return model
