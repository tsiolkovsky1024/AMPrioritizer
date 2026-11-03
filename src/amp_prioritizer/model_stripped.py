"""Compact predictor with AMP, species and MIC heads only.

Mirrors amp_prioritizer.model:AMPModel but drops physchem_head + hemolysis_head and the
Stage 2b `pooled` SupCon path is unused (the loss never reads `pooled` here).
The species_encoder + species_head + species_mic_head are kept verbatim because
they drive the per-species MIC regression and per-species classification metrics
reported by the benchmark.

LoRA path is kept for completeness but the trainer config sets `finetune_mode: full`
to use the merged Stage 0 backbone produced by merge_stage0_lora.py.
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

from amp_prioritizer.species_conditioning import (
    LEGACY_KMER_DIM,
    SpeciesFeatureEncoder,
    SpeciesFusion,
    load_species_feature_tensor,
)


KMER_DIM = LEGACY_KMER_DIM


class MaskedMeanPool(nn.Module):
    def forward(self, hidden, mask):
        m = mask.unsqueeze(-1).to(hidden.dtype)
        s = (hidden * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return s / denom


class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class KMerSpeciesEncoder(nn.Module):
    def __init__(self, kmer_features, hidden, mlp_hidden=512):
        super().__init__()
        assert kmer_features.ndim == 2 and kmer_features.shape[1] == KMER_DIM, kmer_features.shape
        self.register_buffer("features", kmer_features.float().contiguous())
        self.mlp = nn.Sequential(
            nn.Linear(KMER_DIM, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    @property
    def n_species(self):
        return int(self.features.shape[0])

    def forward(self):
        return self.mlp(self.features)


def _load_kmer_features_as_tensor(path, species_list):
    if species_list is None:
        raise ValueError("species_list is required when type=kmer_mlp")
    data = np.load(path, allow_pickle=True)
    names = list(data["names"])
    features = data["features"]
    if features.shape[1] != KMER_DIM:
        raise ValueError(f"{path}: expected width {KMER_DIM}, got {features.shape[1]}")
    name_to_idx = {n: i for i, n in enumerate(names)}
    missing = [s for s in species_list if s not in name_to_idx]
    if missing:
        raise KeyError(f"{path} missing species: {missing[:5]}")
    rows = np.stack([features[name_to_idx[s]] for s in species_list])
    return torch.from_numpy(rows.astype(np.float32))


class AMPStrippedModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        *,
        n_species,
        finetune_mode="full",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        head_hidden=512,
        head_dropout=0.2,
        gradient_checkpointing=True,
        hf_cache_dir=None,
        species_encoder_cfg=None,
        species_list=None,
        species_fusion="concat",
        mic_head_mode="distribution",
    ):
        super().__init__()
        kwargs = {"cache_dir": hf_cache_dir} if hf_cache_dir else {}
        # backbone_name can be either a HF id (e.g. facebook/esm2_t33_650M_UR50D)
        # or a local merged dir produced by merge_stage0_lora.py.
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

        enc_cfg = dict(species_encoder_cfg) if species_encoder_cfg else {"type": "embedding"}
        enc_cfg.setdefault("type", "embedding")
        self.species_encoder_type = enc_cfg["type"]
        if self.species_encoder_type == "kmer_mlp":
            path = enc_cfg.get("kmer_features_path")
            if not path or not os.path.exists(path):
                print(f"[AMPStrippedModel] WARNING: kmer_features_path missing "
                      f"({path!r}); falling back to embedding encoder")
                self.species_encoder_type = "embedding"
                self.species_encoder = None
                self.species_emb = nn.Embedding(self.n_species, hidden)
            else:
                feats = load_species_feature_tensor(path, species_list, expected_dim=KMER_DIM)
                if feats.shape[0] != self.n_species:
                    raise ValueError(f"kmer features cover {feats.shape[0]} species but "
                                     f"n_species={self.n_species}")
                self.species_encoder = SpeciesFeatureEncoder(
                    feats, hidden,
                    mlp_hidden=enc_cfg.get("mlp_hidden", 512),
                    dropout=enc_cfg.get("dropout", 0.0),
                )
                self.species_emb = None
        elif self.species_encoder_type == "embedding":
            self.species_encoder = None
            self.species_emb = nn.Embedding(self.n_species, hidden)
        else:
            raise ValueError(
                "species_encoder.type must be embedding or kmer_mlp"
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

    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids, attention_mask):
        pooled = self.encode(input_ids, attention_mask)
        amp_logit = self.amp_head(pooled).squeeze(-1)

        if self.species_encoder is not None:
            sp_emb = self.species_encoder()
            N = sp_emb.shape[0]
        else:
            sp_idx = torch.arange(self.n_species, device=pooled.device)
            sp_emb = self.species_emb(sp_idx)
            N = self.n_species
        feat = self.species_fusion(pooled, sp_emb)

        species_logit = self.species_head(feat).squeeze(-1)
        mic_out = self.species_mic_head(feat)
        mic_mu = mic_out[..., 0]
        mic_point = mic_mu
        if self.mic_head_mode == "dual_residual":
            mic_point = (
                self.mic_global_head(pooled).expand(-1, N)
                + self.mic_residual_head(feat).squeeze(-1)
            )
        return {
            "amp_logit": amp_logit,
            "species_logit": species_logit,
            "mic_mu": mic_mu,
            "mic_log_sigma": mic_out[..., 1],
            "mic_point": mic_point,
            "pooled": pooled,
        }

    def trainable_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable, total

    @staticmethod
    def from_checkpoint(ckpt_path, backbone_name, **kw):
        state = torch.load(ckpt_path, map_location="cpu") if os.path.exists(ckpt_path) else None
        if state is not None:
            kw["species_fusion"] = state.get("species_fusion", kw.get("species_fusion", "concat"))
            kw["mic_head_mode"] = state.get("mic_head_mode", kw.get("mic_head_mode", "distribution"))
        model = AMPStrippedModel(backbone_name, **kw)
        if not os.path.exists(ckpt_path):
            print(f"[from_checkpoint] {ckpt_path} missing; skipping warm-start")
            return model
        assert state is not None
        sd = state.get("model", state)
        new_sd = model.state_dict()
        for k in list(sd.keys()):
            if k in new_sd and sd[k].shape != new_sd[k].shape:
                print(f"[from_checkpoint] shape mismatch {k}: ckpt={tuple(sd[k].shape)} "
                      f"new={tuple(new_sd[k].shape)} — skipping")
                sd.pop(k)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[from_checkpoint] {len(missing)} missing keys (defaults kept)")
        if unexpected:
            print(f"[from_checkpoint] {len(unexpected)} unexpected keys (dropped)")
        return model
