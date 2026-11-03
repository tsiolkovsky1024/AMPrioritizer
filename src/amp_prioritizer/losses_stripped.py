"""Stripped multi-task loss — only AMP/sp/MIC terms.

L = L_amp                         (BCE, label_smoothing)
  + α_sp   · L_sp                 (per-species activity BCE on -1/0/1 labels)
  + α_cens · L_censored           (Tobit/censored Gaussian NLL on MIC)
  + α_hub  · L_huber              (Huber on active-only MIC)
  + α_λ    · L_lambda             (LambdaRank-lite, rank loss)
  + α_lmle · L_listmle            (ListMLE, rank loss)

Drops L_physchem, L_hemo, L_selectivity and L_consistency from amp_prioritizer.losses.
Reuses the underlying primitives from amp_prioritizer.losses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


from amp_prioritizer.losses import (
    bce_with_smoothing,
    censored_gaussian_nll,
    masked_huber,
    masked_interval_pairwise_rank_loss,
    masked_lambda_ndcg_loss,
    masked_pooled_lambda_ndcg_loss,
    masked_listmle_loss,
    masked_pairwise_lambda_rank_loss,
    ranking_pair_counts,
)


def stripped_loss(
    amp_logit: torch.Tensor,
    species_logit: torch.Tensor,
    mic_mu: torch.Tensor,
    mic_log_sigma: torch.Tensor,
    amp_target: torch.Tensor,
    species_target: torch.Tensor,
    mic_target: torch.Tensor,
    mic_mask: torch.Tensor,
    mic_censored_right: torch.Tensor,
    mic_censored_left: torch.Tensor | None = None,
    *,
    mic_point: torch.Tensor | None = None,
    rank_pred: torch.Tensor | None = None,
    rank_target: torch.Tensor | None = None,
    rank_mask: torch.Tensor | None = None,
    rank_censored_right: torch.Tensor | None = None,
    rank_censored_left: torch.Tensor | None = None,
    alpha_amp: float = 1.0,
    alpha_species: float = 0.5,
    alpha_censored: float = 0.3,
    alpha_huber_active: float = 0.2,
    alpha_lambda: float = 0.3,
    alpha_listmle: float = 0.2,
    alpha_rank_exact: float = 0.0,
    alpha_rank_censored: float = 0.0,
    alpha_rank_pooled: float = 0.0,
    label_smoothing: float = 0.1,
    huber_delta: float = 1.0,
    species_reduction: str = "cell_mean",
    species_weights: torch.Tensor | None = None,
    censoring_mode: str = "proper",
    censor_event_multiplier: float = 1.0,
    alpha_point_consistency: float = 0.0,
    rank_loss_type: str = "gap_pairwise",
    rank_min_gap_log10: float = 0.0,
    rank_ndcg_k: int = 50,
    listmle_top_weighted: bool = False,
) -> tuple[torch.Tensor, dict]:
    l_amp = bce_with_smoothing(amp_logit, amp_target, smoothing=label_smoothing)
    l_sp = bce_with_smoothing(
        species_logit.reshape(-1),
        species_target.reshape(-1),
        smoothing=label_smoothing,
    )
    if mic_censored_left is None:
        mic_censored_left = torch.zeros_like(mic_mask)
    if censoring_mode == "proper":
        loss_mask = mic_mask
        loss_right = mic_censored_right
        loss_left = mic_censored_left
    elif censoring_mode == "exact":
        loss_mask = mic_mask
        loss_right = torch.zeros_like(mic_censored_right)
        loss_left = torch.zeros_like(mic_censored_left)
    elif censoring_mode == "drop":
        loss_mask = mic_mask & ~mic_censored_right.bool() & ~mic_censored_left.bool()
        loss_right = torch.zeros_like(mic_censored_right)
        loss_left = torch.zeros_like(mic_censored_left)
    else:
        raise ValueError(f"unsupported censoring_mode: {censoring_mode!r}")

    l_cens, mic_parts = censored_gaussian_nll(
        mic_mu, mic_log_sigma, mic_target, loss_mask,
        censored_right=loss_right,
        censored_left=loss_left,
        reduction=species_reduction,
        species_weights=species_weights,
        censor_event_multiplier=censor_event_multiplier,
        return_components=True,
    )

    active_mask = loss_mask & ~loss_right.bool() & ~loss_left.bool()
    point = mic_mu if mic_point is None else mic_point
    l_huber = masked_huber(
        point, mic_target, active_mask, delta=huber_delta,
        reduction=species_reduction, species_weights=species_weights,
    )
    rank_point = point if rank_pred is None else rank_pred
    rank_truth = mic_target if rank_target is None else rank_target
    rank_observed = loss_mask if rank_mask is None else rank_mask.bool()
    rank_right = loss_right if rank_censored_right is None else rank_censored_right.bool()
    rank_left = loss_left if rank_censored_left is None else rank_censored_left.bool()
    rank_exact_mask = rank_observed & ~rank_right & ~rank_left
    l_rank_exact = point.new_zeros(())
    l_rank_censored = point.new_zeros(())
    l_rank_pooled = point.new_zeros(())
    pair_counts = (0, 0, 0)
    if rank_loss_type == "none":
        l_lambda = point.new_zeros(())
        l_listmle = point.new_zeros(())
    elif rank_loss_type == "global_v2":
        l_lambda = point.new_zeros(())
        l_listmle = point.new_zeros(())
        if alpha_rank_exact:
            l_rank_exact = masked_lambda_ndcg_loss(
                rank_point, rank_truth, rank_exact_mask, k=rank_ndcg_k,
                min_target_gap=rank_min_gap_log10,
            )
        if alpha_rank_censored:
            l_rank_censored = masked_interval_pairwise_rank_loss(
                rank_point, rank_truth, rank_observed, rank_right, rank_left,
                min_target_gap=rank_min_gap_log10,
            )
        if alpha_rank_pooled:
            l_rank_pooled = masked_pooled_lambda_ndcg_loss(
                rank_point, rank_truth, rank_exact_mask, k=rank_ndcg_k,
                min_target_gap=rank_min_gap_log10,
            )
        pair_counts = ranking_pair_counts(
            rank_truth, rank_observed, rank_right, rank_left,
            min_target_gap=rank_min_gap_log10,
        )
    elif rank_loss_type == "gap_pairwise":
        l_lambda = masked_pairwise_lambda_rank_loss(
            point, mic_target, active_mask, min_target_gap=rank_min_gap_log10,
        )
        l_listmle = masked_listmle_loss(
            point, mic_target, active_mask, top_weighted=listmle_top_weighted,
        )
    elif rank_loss_type == "listmle":
        l_lambda = point.new_zeros(())
        l_listmle = masked_listmle_loss(
            point, mic_target, active_mask, top_weighted=listmle_top_weighted,
        )
    elif rank_loss_type == "lambda_ndcg":
        l_lambda = masked_lambda_ndcg_loss(
            point, mic_target, active_mask, k=rank_ndcg_k,
            min_target_gap=rank_min_gap_log10,
        )
        l_listmle = masked_listmle_loss(
            point, mic_target, active_mask, top_weighted=listmle_top_weighted,
        )
    else:
        raise ValueError(f"unsupported rank_loss_type: {rank_loss_type!r}")
    l_point_consistency = (
        F.smooth_l1_loss(point[active_mask], mic_mu[active_mask])
        if active_mask.any() else mic_mu.new_zeros(())
    )

    total = (alpha_amp * l_amp
             + alpha_species * l_sp
             + alpha_censored * l_cens
             + alpha_huber_active * l_huber
             + alpha_lambda * l_lambda
             + alpha_listmle * l_listmle
             + alpha_rank_exact * l_rank_exact
             + alpha_rank_censored * l_rank_censored
             + alpha_rank_pooled * l_rank_pooled
             + alpha_point_consistency * l_point_consistency)

    parts = {
        "loss_amp": l_amp.detach(),
        "loss_sp": l_sp.detach(),
        "loss_censored": l_cens.detach(),
        "loss_mic_exact": mic_parts["exact"].detach(),
        "loss_mic_censored": mic_parts["censored"].detach(),
        "mic_exact_count": mic_parts["n_exact"].detach(),
        "mic_censored_count": mic_parts["n_censored"].detach(),
        "loss_huber_active": l_huber.detach(),
        "loss_lambda": l_lambda.detach(),
        "loss_listmle": l_listmle.detach(),
        "loss_rank_exact": l_rank_exact.detach(),
        "loss_rank_censored": l_rank_censored.detach(),
        "loss_rank_pooled": l_rank_pooled.detach(),
        "rank_exact_pairs": point.new_tensor(pair_counts[0]),
        "rank_censored_pairs": point.new_tensor(pair_counts[1]),
        "rank_pooled_pairs": point.new_tensor(pair_counts[2]),
        "rank_nonzero": point.new_tensor(float(any(pair_counts))),
        "loss_point_consistency": l_point_consistency.detach(),
        "loss": total.detach(),
    }
    return total, parts
