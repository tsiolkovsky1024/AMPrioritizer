"""ARDecoder — 4-layer causal transformer + cross-attention to (ĥ, c_emb).

Reuses the ESM-2 tokenizer vocabulary (33 tokens: 20 standard AA + B/Z/U/O/X + cls/eos/pad/mask/...).

Inputs:
  input_ids:    [B, T]  AA token sequence (including BOS=cls, EOS=eos)
  h_hat:        [B, d_pool] decoded pooled embedding from LatentCVAE
  c:            [B, d_c]    condition vector (from ConditionEncoder)
Output:
  logits:       [B, T, V]  next-token logits

Sampling:
  generate(h_hat, c, length, temperature, top_p)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        pooled_dim: int = 1280,
        d_c: int = 39,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 200,
    ):
        super().__init__()
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(max_len + 2, d_model)
        # memory: 2 tokens (ĥ projection, c projection)
        self.h_proj = nn.Linear(pooled_dim, d_model)
        self.c_proj = nn.Linear(d_c, d_model)
        self.mem_norm = nn.LayerNorm(d_model)
        self.tok_norm = nn.LayerNorm(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _build_memory(self, h_hat: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # memory: [B, 2, d_model]
        h_tok = self.h_proj(h_hat).unsqueeze(1)
        c_tok = self.c_proj(c).unsqueeze(1)
        return self.mem_norm(torch.cat([h_tok, c_tok], dim=1))

    def forward(self, input_ids: torch.Tensor, h_hat: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        tgt = self.tok_norm(self.tok_emb(input_ids) + self.pos_emb(pos))
        memory = self._build_memory(h_hat, c)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T,
                                                                  device=input_ids.device)
        tgt_kpm = (input_ids == self.pad_token_id)
        out = self.decoder(tgt, memory,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_kpm)
        return self.lm_head(out)

    @torch.no_grad()
    def generate(
        self,
        h_hat: torch.Tensor,
        c: torch.Tensor,
        target_length: torch.Tensor,        # [B] long, exact decoded length (excl BOS/EOS)
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Sample [B, T] tokens (including BOS, EOS). target_length is the AA-only length.

        Output shape: [B, max(target_length)+2].
        """
        B = h_hat.shape[0]
        L = int(target_length.max().item()) + 1   # +1 for BOS  (EOS added at end)
        L = min(L, self.max_len + 1)
        device = h_hat.device

        toks = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        for step in range(L):
            logits = self.forward(toks, h_hat, c)[:, -1, :] / max(temperature, 1e-6)
            # ban BOS / PAD / EOS until target_length reached
            ban = (toks.shape[1] - 1) < target_length        # still need more tokens
            logits[:, self.pad_token_id] = -1e9
            logits[:, self.bos_token_id] = -1e9
            ban_eos_mask = ban.unsqueeze(-1)
            logits = torch.where(ban_eos_mask & (torch.arange(logits.shape[-1], device=device) == self.eos_token_id),
                                 torch.full_like(logits, -1e9), logits)
            # top-p
            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, dim=-1, descending=True)
                cumprob = torch.softmax(sorted_l, dim=-1).cumsum(dim=-1)
                cutoff = (cumprob > top_p).float().argmax(dim=-1, keepdim=True)
                mask_topp = torch.zeros_like(sorted_l, dtype=torch.bool)
                mask_topp.scatter_(1, cutoff, True)
                cum_mask = mask_topp.cumsum(dim=-1).bool()
                sorted_l = sorted_l.masked_fill(cum_mask, -1e9)
                # scatter back
                logits = torch.empty_like(logits).scatter_(1, sorted_i, sorted_l)

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, self.pad_token_id), next_tok)
            toks = torch.cat([toks, next_tok.unsqueeze(-1)], dim=-1)
            finished = finished | (next_tok == self.eos_token_id)
            # force EOS at target_length
            at_target = (toks.shape[1] - 1) == target_length
            if at_target.any():
                toks[at_target & ~finished, -1] = self.eos_token_id
                finished = finished | at_target
            if finished.all():
                break
        return toks
