"""Generator — wrapper around LatentCVAE + LengthHead + ARDecoder.

forward(h_real, seqs_ids, species_onehot, length_bucket, charge_bucket)
  returns dict with cvae_out / length_logits / lm_logits — all of them used for training.

sample(species_onehot, length_bucket=None, charge_bucket=None, n=1, temperature=1.0)
  → list[str]: complete AA sequence strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


from amp_prioritizer.gen.cvae import (
    CVAEConfig, ConditionEncoder, LatentCVAE, LengthHead, LENGTH_BUCKETS,
    bucket_charge, bucket_length,
)
from amp_prioritizer.gen.ar_decoder import ARDecoder


class Generator(nn.Module):
    def __init__(self, tokenizer, cvae_cfg: CVAEConfig | None = None,
                 ar_d_model: int = 512, ar_n_layers: int = 4):
        super().__init__()
        self.tokenizer = tokenizer
        cvae_cfg = cvae_cfg or CVAEConfig()
        self.cvae_cfg = cvae_cfg
        self.cvae = LatentCVAE(cvae_cfg)
        d_c = self.cvae.cond.d_c
        self.length_head = LengthHead(cvae_cfg.pooled_dim, d_c)
        self.ar = ARDecoder(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            pooled_dim=cvae_cfg.pooled_dim, d_c=d_c,
            d_model=ar_d_model, n_layers=ar_n_layers,
        )

    def build_condition(self, species_onehot: torch.Tensor,
                        length_bucket: torch.Tensor,
                        charge_bucket: torch.Tensor) -> torch.Tensor:
        return self.cvae.cond(species_onehot, length_bucket, charge_bucket)

    def forward(self, h_real: torch.Tensor, input_ids: torch.Tensor,
                species_onehot: torch.Tensor, length_bucket: torch.Tensor,
                charge_bucket: torch.Tensor) -> dict:
        c = self.build_condition(species_onehot, length_bucket, charge_bucket)
        cvae_out = self.cvae(h_real, c)
        h_hat = cvae_out["h_hat"]
        length_logits = self.length_head(h_hat.detach(), c)        # detach to stop AR gradient through length
        lm_logits = self.ar(input_ids, h_hat, c)
        return {**cvae_out, "c": c, "length_logits": length_logits, "lm_logits": lm_logits}

    @torch.no_grad()
    def sample(
        self,
        species_onehot: torch.Tensor,
        length_bucket: torch.Tensor | None = None,
        charge_bucket: torch.Tensor | None = None,
        n: int = 1, temperature: float = 1.0, top_p: float = 0.95,
        use_predicted_length: bool = True,
    ) -> tuple[list[str], dict]:
        device = next(self.parameters()).device
        if species_onehot.dim() == 1:
            species_onehot = species_onehot.unsqueeze(0)
        species_onehot = species_onehot.to(device)
        # Default charge bucket = middle (2: 0~2 charge)
        if charge_bucket is None:
            charge_bucket = torch.full((species_onehot.shape[0],), 2,
                                       dtype=torch.long, device=device)
        if length_bucket is None:
            length_bucket = torch.full((species_onehot.shape[0],), 4,  # bucket 4 = 24 aa
                                       dtype=torch.long, device=device)
        if n > 1:
            species_onehot = species_onehot.repeat_interleave(n, dim=0)
            length_bucket = length_bucket.repeat_interleave(n, dim=0)
            charge_bucket = charge_bucket.repeat_interleave(n, dim=0)
        c = self.build_condition(species_onehot, length_bucket, charge_bucket)
        z, h_hat = self.cvae.sample_latent(c, n=c.shape[0], temperature=temperature)

        if use_predicted_length:
            len_logits = self.length_head(h_hat, c)
            len_idx = len_logits.argmax(dim=-1)
            target_len = torch.tensor([LENGTH_BUCKETS[i.item()] for i in len_idx],
                                      dtype=torch.long, device=device)
        else:
            target_len = torch.tensor([LENGTH_BUCKETS[i.item()] for i in length_bucket],
                                      dtype=torch.long, device=device)

        toks = self.ar.generate(h_hat, c, target_len, temperature=temperature, top_p=top_p)
        seqs: list[str] = []
        for row in toks.tolist():
            # Drop BOS, cut at EOS, convert to AA characters
            chars = []
            for t in row[1:]:
                if t == self.ar.eos_token_id or t == self.ar.pad_token_id:
                    break
                tok_str = self.tokenizer.convert_ids_to_tokens(t)
                if tok_str and len(tok_str) == 1 and tok_str.isalpha():
                    chars.append(tok_str)
            seqs.append("".join(chars))
        return seqs, {"length": target_len.cpu().tolist(), "h_hat": h_hat}
