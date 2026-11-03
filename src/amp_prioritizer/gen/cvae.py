"""LatentCVAE — conditional VAE over the ESM-2 pooled embedding space.

Design:
  - encoder q_φ(z | h, c):  MLP 1280 → 512 → 2·d_z (outputs μ, log σ)
  - prior    p_ψ(z | c):    FiLM-condition c → 2·d_z (μ_c, log σ_c). Gaussian.
  - decoder  p_θ(ĥ | z, c): MLP [z ⊕ c] → 512 → 1280
  - condition c: species one-hot (7) ⊕ length-bucket emb (16) ⊕ charge-bucket emb (16) → 39
  - ELBO = E_q[log p(h|z,c)] − β·KL(q(z|h,c) || p(z|c))
    log p(h|z,c) = −0.5·||h − ĥ||² / σ_recon²  (fixed σ_recon=1)
  - β cyclic-annealed (β_schedule) + free-bits floor (prevents KL collapse).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Length buckets
LENGTH_BUCKETS = (8, 12, 16, 20, 24, 28, 32, 40, 50, 64, 80, 100)


def bucket_length(length: int | torch.Tensor) -> torch.Tensor:
    """Map a length to the nearest bucket index in LENGTH_BUCKETS."""
    if isinstance(length, int):
        length = torch.tensor([length])
    buckets = torch.tensor(LENGTH_BUCKETS, device=length.device, dtype=length.dtype)
    diffs = (length.unsqueeze(-1) - buckets.unsqueeze(0)).abs()
    return diffs.argmin(dim=-1)


def bucket_charge(charge: float | torch.Tensor) -> torch.Tensor:
    """5 charge buckets: <-2, -2~0, 0~2, 2~5, >5"""
    if isinstance(charge, (int, float)):
        charge = torch.tensor([float(charge)])
    edges = torch.tensor([-2.0, 0.0, 2.0, 5.0], device=charge.device, dtype=charge.dtype)
    return (charge.unsqueeze(-1) >= edges.unsqueeze(0)).sum(dim=-1)


@dataclass
class CVAEConfig:
    pooled_dim: int = 1280                # ESM-2 t33 650M hidden
    z_dim: int = 128
    hidden_dim: int = 512
    n_species: int = 20                   # v3 top-20; override from load_latest_species()
    length_emb_dim: int = 16
    charge_emb_dim: int = 16
    n_charge_buckets: int = 5
    dropout: float = 0.1


class ConditionEncoder(nn.Module):
    """species one-hot + length-bucket emb + charge-bucket emb → d_c"""
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.length_emb = nn.Embedding(len(LENGTH_BUCKETS), cfg.length_emb_dim)
        self.charge_emb = nn.Embedding(cfg.n_charge_buckets, cfg.charge_emb_dim)
        self.d_c = cfg.n_species + cfg.length_emb_dim + cfg.charge_emb_dim

    def forward(self, species_onehot: torch.Tensor,
                length_bucket: torch.Tensor,
                charge_bucket: torch.Tensor) -> torch.Tensor:
        # species_onehot: [B, n_species]  length_bucket: [B] long  charge_bucket: [B] long
        le = self.length_emb(length_bucket)
        ce = self.charge_emb(charge_bucket)
        return torch.cat([species_onehot, le, ce], dim=-1)


class LatentCVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.cond = ConditionEncoder(cfg)
        d_c = self.cond.d_c
        # q(z | h, c)
        self.q_net = nn.Sequential(
            nn.Linear(cfg.pooled_dim + d_c, cfg.hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 2 * cfg.z_dim),
        )
        # p(z | c)
        self.p_net = nn.Sequential(
            nn.Linear(d_c, cfg.hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 2 * cfg.z_dim),
        )
        # decoder p(ĥ | z, c)
        self.dec = nn.Sequential(
            nn.Linear(cfg.z_dim + d_c, cfg.hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.pooled_dim),
        )

    def encode(self, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_params = self.q_net(torch.cat([h, c], dim=-1))
        mu, log_sigma = z_params.chunk(2, dim=-1)
        return mu, log_sigma.clamp(min=-5.0, max=5.0)

    def prior(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_params = self.p_net(c)
        mu, log_sigma = z_params.chunk(2, dim=-1)
        return mu, log_sigma.clamp(min=-5.0, max=5.0)

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, c], dim=-1))

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> dict:
        mu_q, ls_q = self.encode(h, c)
        mu_p, ls_p = self.prior(c)
        z = self.reparameterize(mu_q, ls_q)
        h_hat = self.decode(z, c)
        return {"mu_q": mu_q, "log_sigma_q": ls_q,
                "mu_p": mu_p, "log_sigma_p": ls_p,
                "z": z, "h_hat": h_hat}

    @torch.no_grad()
    def sample_latent(self, c: torch.Tensor, n: int = 1, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n latents z from the prior p(z|c), returns (z, h_hat)."""
        if c.dim() == 1:
            c = c.unsqueeze(0)
        if n > 1:
            c = c.expand(n, -1)
        mu_p, ls_p = self.prior(c)
        std = torch.exp(ls_p) * temperature
        z = mu_p + std * torch.randn_like(std)
        h_hat = self.decode(z, c)
        return z, h_hat


def kl_gaussian(mu_q: torch.Tensor, log_sigma_q: torch.Tensor,
                mu_p: torch.Tensor, log_sigma_p: torch.Tensor,
                free_bits: float = 0.0) -> torch.Tensor:
    """KL(q || p) for two diagonal Gaussians, per-dim free bits.
    Returns a [B] scalar (summed over z dims; averaging over the batch is left to the caller).
    """
    var_q = torch.exp(2 * log_sigma_q)
    var_p = torch.exp(2 * log_sigma_p)
    kl_per_dim = (log_sigma_p - log_sigma_q
                  + 0.5 * (var_q + (mu_q - mu_p) ** 2) / var_p
                  - 0.5)
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    return kl_per_dim.sum(dim=-1)


class LengthHead(nn.Module):
    """[ĥ ⊕ c] → softmax over LENGTH_BUCKETS (12-way classification)."""
    def __init__(self, pooled_dim: int, d_c: int, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pooled_dim + d_c, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(LENGTH_BUCKETS)),
        )

    def forward(self, h_hat: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h_hat, c], dim=-1))     # [B, 12] logits


def beta_schedule(step: int, period: int = 10000, max_beta: float = 0.5) -> float:
    """Cyclic β annealing for CVAE — prevents posterior collapse.

    One full cycle: first 1/3 β=0, middle 1/3 rises linearly to max_beta, last 1/3 holds max_beta.
    """
    phase = (step % period) / period
    if phase < 1 / 3:
        return 0.0
    if phase < 2 / 3:
        return max_beta * 3.0 * (phase - 1 / 3)
    return max_beta
