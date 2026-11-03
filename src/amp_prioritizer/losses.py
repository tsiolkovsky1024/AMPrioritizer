"""Multi-task losses for AMP v3.

Covers:
  - bce_with_smoothing          BCE on cls heads, ignores -1, symmetric label smoothing
  - masked_huber                Huber on active MIC subset
  - masked_pairwise_lambda_rank Gap-weighted RankNet per species column
  - masked_lambda_ndcg_loss      Pairwise LambdaRank with true Delta-NDCG weights
  - masked_listmle_loss         ListMLE (Plackett-Luce NLL), top-of-tail dominates gradient
  - censored_gaussian_nll       Tobit-style NLL with right- and left-censoring (per-row τ)
  - hemolysis_censored_nll      same, per-sequence HC50 (log10 µM)
  - selectivity_contrastive     SupCon over species-activity profiles
  - consistency_kl              symmetric KL of two dropout forwards' mic_mu softmax
  - multi_task_loss             assembles the above; stage ∈ {'2a', '2b'}

Why both MSE/Huber and censored NLL? Huber handles the active subset (where
log10_mic is a real number); the censored NLL handles right/left-censored
measurements without dragging μ to the reporting wall.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Classification / regression primitives
# ---------------------------------------------------------------------------

def bce_with_smoothing(
    logits: torch.Tensor, target: torch.Tensor,
    *, ignore_index: int = -1, smoothing: float = 0.1,
) -> torch.Tensor:
    mask = target != ignore_index
    if mask.sum() == 0:
        return logits.new_zeros(())
    t = target[mask].to(logits.dtype)
    if smoothing > 0:
        t = t * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits[mask], t)


def masked_huber(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, delta: float = 1.0, reduction: str = "cell_mean",
    species_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return pred.new_zeros(())
    elementwise = F.huber_loss(
        pred, target.to(pred.dtype), delta=delta, reduction="none",
    )
    return masked_species_reduce(
        elementwise, mask, reduction=reduction,
        species_weights=species_weights,
    )


def masked_species_reduce(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "cell_mean",
    species_weights: torch.Tensor | None = None,
    observation_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce a [batch, species] loss without letting large species dominate."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if values.shape != mask.shape:
        raise ValueError(f"values/mask shape mismatch: {values.shape} vs {mask.shape}")
    if not mask.any():
        return values.new_zeros(())
    if observation_weights is None:
        observation_weights = torch.ones_like(values)
    elif observation_weights.shape != values.shape:
        raise ValueError("observation_weights must have the same shape as values")
    else:
        observation_weights = observation_weights.to(values.device, values.dtype)
    if not torch.isfinite(observation_weights).all() or (observation_weights < 0).any():
        raise ValueError("observation_weights must be finite and non-negative")
    weighted_mask = observation_weights * mask.to(values.dtype)
    if weighted_mask.sum() <= 0:
        return values.new_zeros(())
    if reduction == "cell_mean":
        return (values * weighted_mask).sum() / weighted_mask.sum()
    if reduction not in {"species_macro", "inverse_sqrt"}:
        raise ValueError(f"unsupported species reduction: {reduction!r}")

    counts = weighted_mask.sum(dim=0)
    valid = counts > 0
    per_species = (
        (values * weighted_mask).sum(dim=0)[valid]
        / counts[valid]
    )
    if reduction == "species_macro":
        return per_species.mean()

    if species_weights is None:
        weights = counts[valid].to(values.dtype).rsqrt()
    else:
        if species_weights.ndim != 1 or species_weights.numel() != mask.shape[1]:
            raise ValueError("species_weights must have one entry per species")
        weights = species_weights.to(values.device, values.dtype)[valid]
    weights = weights / weights.sum().clamp(min=torch.finfo(values.dtype).eps)
    return (per_species * weights).sum()


def masked_pairwise_lambda_rank_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, min_target_gap: float = 0.0,
) -> torch.Tensor:
    """LambdaRank-lite: per species column, sum |Δt| · softplus(s_i - s_j) over
    ordered pairs (t_i < t_j). `target` is log10(µM); lower = more potent."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    losses: list[torch.Tensor] = []
    for c in range(pred.shape[1]):
        col_mask = mask[:, c]
        if int(col_mask.sum()) < 2:
            continue
        s = pred[col_mask, c]
        t = target[col_mask, c].to(s.dtype)
        if torch.std(t) < 1e-6:
            continue
        pair_gap = t.unsqueeze(0) - t.unsqueeze(1)
        pair_mask = pair_gap > min_target_gap
        if not pair_mask.any():
            continue
        diff = s.unsqueeze(1) - s.unsqueeze(0)
        gap = (t.unsqueeze(0) - t.unsqueeze(1)).abs().detach()
        losses.append((gap * F.softplus(diff))[pair_mask].mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def masked_lambda_ndcg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int = 50,
    min_target_gap: float = 0.0,
) -> torch.Tensor:
    """Pairwise rank loss weighted by predicted-position Delta-NDCG@k."""
    if k < 1:
        raise ValueError("k must be positive")
    if min_target_gap < 0:
        raise ValueError("min_target_gap must be non-negative")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    losses: list[torch.Tensor] = []
    for column in range(pred.shape[1]):
        col_mask = mask[:, column]
        if int(col_mask.sum()) < 2:
            continue
        scores = pred[col_mask, column]
        truth = target[col_mask, column].to(scores.dtype)
        if torch.std(truth) < 1e-6:
            continue
        pair_gap = truth.unsqueeze(0) - truth.unsqueeze(1)
        pair_mask = pair_gap > min_target_gap
        if not pair_mask.any():
            continue

        n = int(scores.numel())
        predicted_order = torch.argsort(scores.detach())
        positions = torch.empty(n, dtype=torch.long, device=scores.device)
        positions[predicted_order] = torch.arange(n, device=scores.device)
        discounts = torch.where(
            positions < k,
            torch.log2(positions.to(scores.dtype) + 2.0).reciprocal(),
            torch.zeros(n, dtype=scores.dtype, device=scores.device),
        )
        gains = torch.pow(scores.new_tensor(2.0), -truth).detach()
        ideal_order = torch.argsort(truth)
        ideal_len = min(k, n)
        ideal_discounts = torch.log2(
            torch.arange(ideal_len, device=scores.device, dtype=scores.dtype) + 2.0
        ).reciprocal()
        ideal_dcg = (gains[ideal_order[:ideal_len]] * ideal_discounts).sum().clamp_min(
            torch.finfo(scores.dtype).eps
        )
        delta = (
            (gains.unsqueeze(1) - gains.unsqueeze(0)).abs()
            * (discounts.unsqueeze(1) - discounts.unsqueeze(0)).abs()
            / ideal_dcg
        ).detach()
        active = pair_mask & (delta > 0)
        if not active.any():
            continue
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        losses.append((delta * F.softplus(score_diff))[active].mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def masked_interval_pairwise_rank_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    censored_right: torch.Tensor, censored_left: torch.Tensor,
    *, min_target_gap: float = 0.0,
) -> torch.Tensor:
    """Rank pairs whose exact/censored MIC intervals cannot overlap."""
    mask = mask.bool()
    right = censored_right.bool() & mask
    left = censored_left.bool() & mask
    exact = mask & ~right & ~left
    losses: list[torch.Tensor] = []
    for column in range(pred.shape[1]):
        observed = mask[:, column]
        if int(observed.sum()) < 2:
            continue
        scores = pred[observed, column]
        truth = target[observed, column].to(scores.dtype)
        col_exact = exact[observed, column]
        col_right = right[observed, column]
        col_left = left[observed, column]
        lower = torch.where(col_left, torch.full_like(truth, float("-inf")), truth)
        upper = torch.where(col_right, torch.full_like(truth, float("inf")), truth)
        guaranteed_gap = lower.unsqueeze(0) - upper.unsqueeze(1)
        ordered = guaranteed_gap > min_target_gap
        ordered &= ~(col_exact.unsqueeze(0) & col_exact.unsqueeze(1))
        if not ordered.any():
            continue
        score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
        weights = guaranteed_gap.clamp_min(0).detach()
        losses.append((weights * F.softplus(score_diff))[ordered].mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def masked_pooled_lambda_ndcg_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, k: int = 50, min_target_gap: float = 0.0,
) -> torch.Tensor:
    """LambdaNDCG over all observed peptide-species cells in the rank pool."""
    return masked_lambda_ndcg_loss(
        pred.reshape(-1, 1), target.reshape(-1, 1), mask.reshape(-1, 1).bool(),
        k=k, min_target_gap=min_target_gap,
    )


def ranking_pair_counts(
    target: torch.Tensor, mask: torch.Tensor,
    censored_right: torch.Tensor, censored_left: torch.Tensor,
    *, min_target_gap: float = 0.0,
) -> tuple[int, int, int]:
    """Return exact, censored-partial-order, and pooled exact pair counts."""
    mask = mask.bool()
    right = censored_right.bool() & mask
    left = censored_left.bool() & mask
    exact = mask & ~right & ~left
    exact_pairs = 0
    censored_pairs = 0
    for column in range(target.shape[1]):
        truth = target[:, column]
        gap = truth.unsqueeze(0) - truth.unsqueeze(1)
        col_exact = exact[:, column]
        exact_pairs += int(((gap > min_target_gap) & col_exact.unsqueeze(0)
                           & col_exact.unsqueeze(1)).sum())
        lower = torch.where(left[:, column], torch.full_like(truth, float("-inf")), truth)
        upper = torch.where(right[:, column], torch.full_like(truth, float("inf")), truth)
        ordered = (lower.unsqueeze(0) - upper.unsqueeze(1)) > min_target_gap
        observed = mask[:, column]
        ordered &= observed.unsqueeze(0) & observed.unsqueeze(1)
        ordered &= ~(col_exact.unsqueeze(0) & col_exact.unsqueeze(1))
        censored_pairs += int(ordered.sum())
    flat_truth = target[exact]
    pooled_pairs = 0 if flat_truth.numel() < 2 else int(
        (flat_truth.unsqueeze(0) - flat_truth.unsqueeze(1) > min_target_gap).sum()
    )
    return exact_pairs, censored_pairs, pooled_pairs


def masked_listmle_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, top_weighted: bool = False,
) -> torch.Tensor:
    """ListMLE (Plackett-Luce NLL) per species column. The first step (true top-1)
    dominates the gradient when the model puts the wrong peptide at rank 1."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    losses: list[torch.Tensor] = []
    for c in range(pred.shape[1]):
        col_mask = mask[:, c]
        if int(col_mask.sum()) < 3:
            continue
        s = pred[col_mask, c]
        t = target[col_mask, c]
        if torch.std(t) < 1e-6:
            continue
        order = torch.argsort(t)
        z = -s[order]
        tail_lse = torch.logcumsumexp(z.flip(0), dim=0).flip(0)
        per_step = z - tail_lse
        if top_weighted:
            weights = torch.log2(
                torch.arange(len(per_step), device=s.device, dtype=s.dtype) + 2.0
            ).reciprocal()
            losses.append(-(weights * per_step).sum() / weights.sum())
        else:
            losses.append(-per_step.mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Censored (Tobit) Gaussian NLL
# ---------------------------------------------------------------------------

def _log_survival(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - Phi(x)), including under mixed precision."""
    if hasattr(torch.special, "log_ndtr"):
        work = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
        return torch.special.log_ndtr(-work).to(x.dtype)
    return torch.log(0.5 * torch.erfc(x / math.sqrt(2.0)).clamp(min=1e-30))


def _log_cdf(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log Phi(x), including under mixed precision."""
    if hasattr(torch.special, "log_ndtr"):
        work = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
        return torch.special.log_ndtr(work).to(x.dtype)
    return torch.log(0.5 * (1 + torch.erf(x / math.sqrt(2.0))).clamp(min=1e-30))


def censored_gaussian_nll(
    mic_mu: torch.Tensor,
    mic_log_sigma: torch.Tensor,
    target: torch.Tensor,
    mic_mask: torch.Tensor,
    censored_right: torch.Tensor,
    censored_left: torch.Tensor | None = None,
    *,
    log_sigma_clamp: tuple[float, float] = (-2.0, 2.0),
    reduction: str = "cell_mean",
    species_weights: torch.Tensor | None = None,
    censor_event_multiplier: float = 1.0,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Tobit Gaussian NLL with per-observation censoring thresholds."""
    if not math.isfinite(censor_event_multiplier) or censor_event_multiplier < 0:
        raise ValueError("censor_event_multiplier must be finite and non-negative")
    if mic_mask.dtype != torch.bool:
        mic_mask = mic_mask.bool()
    if censored_right.dtype != torch.bool:
        censored_right = censored_right.bool()
    if censored_left is None:
        censored_left = torch.zeros_like(mic_mask)
    elif censored_left.dtype != torch.bool:
        censored_left = censored_left.bool()

    log_sigma = mic_log_sigma.clamp(min=log_sigma_clamp[0], max=log_sigma_clamp[1])
    sigma = torch.exp(log_sigma)
    active = mic_mask & ~censored_right & ~censored_left
    right = mic_mask & censored_right
    left = mic_mask & censored_left
    observed = active | right | left
    elementwise = torch.zeros_like(mic_mu)

    if active.any():
        mu_a = mic_mu[active]
        s_a = sigma[active]
        ls_a = log_sigma[active]
        t_a = target[active].to(mu_a.dtype)
        elementwise[active] = 0.5 * ((t_a - mu_a) / s_a).pow(2) + ls_a
    if right.any():
        mu_r = mic_mu[right]
        s_r = sigma[right]
        tau = target[right].to(mu_r.dtype)
        elementwise[right] = -_log_survival((tau - mu_r) / s_r)
    if left.any():
        mu_l = mic_mu[left]
        s_l = sigma[left]
        tau = target[left].to(mu_l.dtype)
        elementwise[left] = -_log_cdf((tau - mu_l) / s_l)

    if not observed.any():
        zero = mic_mu.new_zeros(())
        if return_components:
            return zero, {
                "exact": zero,
                "censored": zero,
                "n_exact": zero,
                "n_censored": zero,
            }
        return zero

    censored = right | left
    observation_weights = torch.ones_like(elementwise)
    observation_weights[censored] = censor_event_multiplier
    loss = masked_species_reduce(
        elementwise,
        observed,
        reduction=reduction,
        species_weights=species_weights,
        observation_weights=observation_weights,
    )
    if not return_components:
        return loss
    components = {
        "exact": masked_species_reduce(
            elementwise,
            active,
            reduction=reduction,
            species_weights=species_weights,
        ),
        "censored": masked_species_reduce(
            elementwise,
            censored,
            reduction=reduction,
            species_weights=species_weights,
        ),
        "n_exact": mic_mu.new_tensor(int(active.sum())),
        "n_censored": mic_mu.new_tensor(int(censored.sum())),
    }
    return loss, components


def hemolysis_censored_nll(
    hemo_mu: torch.Tensor,
    hemo_log_sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    censored_right: torch.Tensor | None = None,
    censored_left: torch.Tensor | None = None,
    *,
    log_sigma_clamp: tuple[float, float] = (-2.0, 2.0),
) -> torch.Tensor:
    """Hemolysis HC50 regression with Tobit censoring. High HC50 = safe; the
    direction is enforced by downstream NarrowScore, not here."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return hemo_mu.new_zeros(())
    if censored_right is None:
        censored_right = torch.zeros_like(mask)
    if censored_left is None:
        censored_left = torch.zeros_like(mask)
    censored_right = censored_right.bool()
    censored_left = censored_left.bool()

    log_sigma = hemo_log_sigma.clamp(min=log_sigma_clamp[0], max=log_sigma_clamp[1])
    sigma = torch.exp(log_sigma)

    active = mask & ~censored_right & ~censored_left
    right = mask & censored_right
    left = mask & censored_left

    parts: list[torch.Tensor] = []
    counts: list[int] = []
    if active.any():
        mu = hemo_mu[active]; s = sigma[active]; ls = log_sigma[active]
        t = target[active].to(mu.dtype)
        parts.append((0.5 * ((t - mu) / s).pow(2) + ls).sum())
        counts.append(int(active.sum()))
    if right.any():
        mu = hemo_mu[right]; s = sigma[right]
        tau = target[right].to(mu.dtype)
        z = (tau - mu) / s
        parts.append((-_log_survival(z)).sum()); counts.append(int(right.sum()))
    if left.any():
        mu = hemo_mu[left]; s = sigma[left]
        tau = target[left].to(mu.dtype)
        z = (tau - mu) / s
        if hasattr(torch.special, "log_ndtr"):
            nll_l = -torch.special.log_ndtr(z)
        else:
            nll_l = -torch.log(0.5 * (1 + torch.erf(z / math.sqrt(2.0))).clamp(min=1e-30))
        parts.append(nll_l.sum()); counts.append(int(left.sum()))

    if not parts:
        return hemo_mu.new_zeros(())
    return torch.stack(parts).sum() / max(1, sum(counts))


# ---------------------------------------------------------------------------
# Selectivity contrastive + consistency KL (Stage 2b)
# ---------------------------------------------------------------------------

def selectivity_contrastive_loss(
    pooled: torch.Tensor,
    species_label: torch.Tensor,
    *,
    temperature: float = 0.1,
    min_known: int = 3,
    pos_threshold: float = 0.7,
) -> torch.Tensor:
    """SupCon over species-activity profiles. An anchor with ≥ min_known known
    species labels and agreement_rate ≥ pos_threshold to another anchor forms a
    positive pair; InfoNCE pulls them together in pooled-embedding space."""
    B, _ = pooled.shape
    if species_label.dtype != torch.long:
        species_label = species_label.long()

    known = species_label != -1
    n_known = known.sum(dim=1)
    valid = n_known >= min_known
    if valid.sum() < 2:
        return pooled.new_zeros(())

    idx = torch.nonzero(valid, as_tuple=True)[0]
    M = idx.numel()
    emb = F.normalize(pooled[idx], dim=-1)
    lbl = species_label[idx]
    msk = known[idx]

    shared = msk.unsqueeze(0) & msk.unsqueeze(1)
    shared_count = shared.sum(dim=-1).clamp(min=1)
    eq = (lbl.unsqueeze(0) == lbl.unsqueeze(1)) & shared
    agree = eq.sum(dim=-1).float() / shared_count.float()
    diag = torch.eye(M, device=pooled.device, dtype=torch.bool)
    has_shared = (shared_count > 0) & ~diag
    pos_mask = (agree >= pos_threshold) & has_shared

    sim = emb @ emb.t() / temperature
    sim = sim.masked_fill(diag, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)

    n_pos = pos_mask.sum(dim=-1)
    has_pos = n_pos > 0
    if has_pos.sum() == 0:
        return pooled.new_zeros(())
    log_prob_safe = log_prob.where(pos_mask, torch.zeros_like(log_prob))
    log_prob_pos = log_prob_safe.sum(dim=-1) / n_pos.clamp(min=1)
    return -log_prob_pos[has_pos].mean()


def consistency_kl_loss(
    mic_mu_1: torch.Tensor,
    mic_mu_2: torch.Tensor,
    mic_mask: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Symmetric KL of two dropout-forward μ softmaxes over species."""
    if mic_mask.dtype != torch.bool:
        mic_mask = mic_mask.bool()
    use = mic_mask.sum(dim=1) >= 2
    if use.sum() == 0:
        return mic_mu_1.new_zeros(())

    neg_inf = torch.finfo(mic_mu_1.dtype).min
    z1 = (mic_mu_1 / temperature).masked_fill(~mic_mask, neg_inf)
    z2 = (mic_mu_2 / temperature).masked_fill(~mic_mask, neg_inf)
    logp1 = F.log_softmax(z1, dim=-1)
    logp2 = F.log_softmax(z2, dim=-1)
    p1 = logp1.exp()
    p2 = logp2.exp()
    kl_12 = (p1 * (logp1 - logp2)).where(mic_mask, torch.zeros_like(logp1)).sum(dim=-1)
    kl_21 = (p2 * (logp2 - logp1)).where(mic_mask, torch.zeros_like(logp1)).sum(dim=-1)
    return (0.5 * (kl_12 + kl_21))[use].mean()


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def multi_task_loss(
    amp_logit: torch.Tensor,
    species_logit: torch.Tensor,
    mic_mu: torch.Tensor,
    mic_log_sigma: torch.Tensor,
    physchem_pred: torch.Tensor,
    pooled: torch.Tensor,
    amp_target: torch.Tensor,
    species_target: torch.Tensor,
    mic_target: torch.Tensor,
    mic_mask: torch.Tensor,
    mic_censored_right: torch.Tensor,
    physchem_target: torch.Tensor,
    *,
    stage: str = "2a",
    mic_mu_2: Optional[torch.Tensor] = None,
    mic_censored_left: Optional[torch.Tensor] = None,
    mic_point: Optional[torch.Tensor] = None,
    hemo_mu: Optional[torch.Tensor] = None,
    hemo_log_sigma: Optional[torch.Tensor] = None,
    hemo_target: Optional[torch.Tensor] = None,
    hemo_mask: Optional[torch.Tensor] = None,
    hemo_censored_right: Optional[torch.Tensor] = None,
    hemo_censored_left: Optional[torch.Tensor] = None,
    alpha_amp: float = 1.0,
    alpha_species: float = 0.5,
    alpha_censored: float = 0.3,
    alpha_huber_active: float = 0.2,
    alpha_lambda: float = 0.3,
    alpha_listmle: float = 0.2,
    alpha_physchem: float = 0.1,
    alpha_hemo: float = 0.2,
    alpha_selectivity: float = 0.05,
    alpha_consistency: float = 0.05,
    label_smoothing: float = 0.1,
    huber_delta: float = 1.0,
    selectivity_temp: float = 0.1,
    consistency_temp: float = 2.0,
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
    """Stage 2a:
        L = L_amp + α_sp·L_sp + α_cens·L_cens + α_huber·L_huber + α_λ·L_λ
            + α_listmle·L_listmle + α_phys·L_phys [+ α_hemo·L_hemo]
    Stage 2b adds: + α_sel·L_selectivity + α_cons·L_consistency
    """
    l_amp = bce_with_smoothing(amp_logit, amp_target, smoothing=label_smoothing)
    l_sp = bce_with_smoothing(species_logit.reshape(-1),
                              species_target.reshape(-1),
                              smoothing=label_smoothing)

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
        censored_right=loss_right, censored_left=loss_left,
        reduction=species_reduction, species_weights=species_weights,
        censor_event_multiplier=censor_event_multiplier,
        return_components=True,
    )

    active_mask = loss_mask & ~loss_right.bool() & ~loss_left.bool()
    point = mic_mu if mic_point is None else mic_point
    l_huber = masked_huber(
        point, mic_target, active_mask, delta=huber_delta,
        reduction=species_reduction, species_weights=species_weights,
    )
    if rank_loss_type == "none":
        l_lambda = point.new_zeros(())
        l_listmle = point.new_zeros(())
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
    l_phys = F.mse_loss(physchem_pred, physchem_target)

    total = (alpha_amp * l_amp
             + alpha_species * l_sp
             + alpha_censored * l_cens
             + alpha_huber_active * l_huber
             + alpha_lambda * l_lambda
             + alpha_listmle * l_listmle
             + alpha_point_consistency * l_point_consistency
             + alpha_physchem * l_phys)

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
        "loss_point_consistency": l_point_consistency.detach(),
        "loss_physchem": l_phys.detach(),
    }

    if (hemo_mu is not None and hemo_log_sigma is not None
            and hemo_target is not None and hemo_mask is not None):
        l_hemo = hemolysis_censored_nll(
            hemo_mu, hemo_log_sigma, hemo_target, hemo_mask,
            censored_right=hemo_censored_right,
            censored_left=hemo_censored_left,
        )
        total = total + alpha_hemo * l_hemo
        parts["loss_hemo"] = l_hemo.detach()

    if stage == "2b":
        l_sel = selectivity_contrastive_loss(pooled, species_target,
                                             temperature=selectivity_temp)
        total = total + alpha_selectivity * l_sel
        parts["loss_selectivity"] = l_sel.detach()
        if mic_mu_2 is not None:
            l_cons = consistency_kl_loss(mic_mu, mic_mu_2, mic_mask,
                                         temperature=consistency_temp)
            total = total + alpha_consistency * l_cons
            parts["loss_consistency"] = l_cons.detach()
        else:
            parts["loss_consistency"] = total.new_zeros(()).detach()

    parts["loss"] = total.detach()
    return total, parts
