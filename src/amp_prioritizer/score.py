"""Narrow-spectrum scoring: combines per-species calibrated scores into a NarrowScore
expressing "high selectivity for species T / no selectivity for species O".

Inputs: a DataFrame with the following columns (from screen.py V2 output):
  - amp_score                (calibrated)
  - sp/<species>             (calibrated per-species activity prob)
  - mic_mu/<species>         (predicted log10(µM))
  - mic_mu_cal/<species>     (optional, isotonic-calibrated μ; used when present)

Formula:
  pMIC_i           = − mic_mu_i  (higher = more potent, − log10 µM)
  p_act_i_cal      = calibrated species activity probability
  amp_cal          = calibrated amp probability
  target_score     = (∏_{i∈T} p_act_i_cal) ^ (1/|T|)        geometric mean
  selectivity      = mean_{i∈T} pMIC_i − mean_{i∈O} pMIC_i

  NarrowScore = amp_cal
              × target_score
              × sigmoid_temp(selectivity − δ)
              × 1[amp_cal ≥ thr_amp & target_score ≥ thr_target]

  Diverse-rerank: − λ_div × max_neighbor_cosine_in_embedding_space (handled by the caller)

Interface:
  narrow_spectrum_score(df, target_species, offtarget_species,
                         thr_amp=0.5, thr_target=0.5, delta=0.5, temp=0.43,
                         strict=True) -> pd.DataFrame   (sorted by NarrowScore descending)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _sigmoid_temp(x: np.ndarray, temp: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / max(temp, 1e-6)))


def narrow_spectrum_score(
    df: pd.DataFrame,
    target_species: Iterable[str],
    offtarget_species: Iterable[str] | None = None,
    *,
    thr_amp: float = 0.5,
    thr_target: float = 0.5,
    delta: float = 0.5,                # at least √10 ≈ 3× selectivity
    temp: float = 0.43,                # 1/log10(10)
    use_calibrated_mic: bool = True,
    strict: bool = True,               # True: samples failing the thresholds get NarrowScore=0
    hemo_penalty: float = 0.5,         # NEW: lower HC50 → larger penalty. 0 = disabled.
    hemo_safe_log10: float = 2.0,      # log10(100µM) — HC50 ≥ this value counts as "safe"
    species_list: Iterable[str] | None = None,  # v3 top-20 species list; required for offtarget inference
) -> pd.DataFrame:
    """Compose narrow-spectrum score with optional hemolysis penalty.

    NarrowScore = amp_cal × target_score × sigmoid_temp(selectivity − δ) × hemo_safety_factor
                  × 1[amp_cal ≥ thr_amp & target_score ≥ thr_target]   (strict mode)

    Hemolysis handling:
      hemo_safety_factor = sigmoid_temp(hemo_mu − hemo_safe_log10)
      hemo_mu is the model-predicted log10(HC50 in µM). High HC50 = safe, factor ≈ 1.
      Column name: hemo_mu (from screen.py output; skipped when absent).
    """
    target_species = list(target_species)
    if not target_species:
        raise ValueError("target_species must not be empty")
    if offtarget_species is None:
        if not species_list:
            raise ValueError(
                "offtarget_species=None requires species_list to be passed as well "
                "(from load_latest_species(data_dir))"
            )
        offtarget_species = [s for s in species_list if s not in target_species]
    else:
        offtarget_species = list(offtarget_species)

    out = df.copy()

    if "amp_score" not in out.columns:
        raise KeyError("df is missing the amp_score column")
    amp_cal = out["amp_score"].to_numpy()

    def _col_or_zero(prefix: str, sp: str, default: float = 0.0) -> np.ndarray:
        col = f"{prefix}/{sp}"
        if col in out.columns:
            return out[col].to_numpy(dtype=np.float64)
        return np.full(len(out), default, dtype=np.float64)

    target_probs = np.stack([_col_or_zero("sp", sp) for sp in target_species], axis=1)
    target_probs = np.clip(target_probs, 1e-6, 1.0)
    target_score = np.exp(np.log(target_probs).mean(axis=1))

    mic_prefix = "mic_mu_cal" if (use_calibrated_mic and
                                  any(c.startswith("mic_mu_cal/") for c in out.columns)
                                  ) else "mic_mu"
    pmic_T = -np.stack([_col_or_zero(mic_prefix, sp, default=4.0) for sp in target_species], axis=1)
    pmic_T_bar = pmic_T.mean(axis=1)
    if offtarget_species:
        pmic_O = -np.stack([_col_or_zero(mic_prefix, sp, default=4.0) for sp in offtarget_species], axis=1)
        pmic_O_bar = pmic_O.mean(axis=1)
        offtarget_probs = np.stack([_col_or_zero("sp", sp) for sp in offtarget_species], axis=1)
        offtarget_probs = np.clip(offtarget_probs, 1e-6, 1.0)
        offtarget_score = np.exp(np.log(offtarget_probs).mean(axis=1))
    else:
        pmic_O_bar = np.zeros(len(out))
        offtarget_score = np.zeros(len(out))

    selectivity = pmic_T_bar - pmic_O_bar
    sel_term = _sigmoid_temp(selectivity - delta, temp)

    # NEW: hemolysis safety factor
    if hemo_penalty > 0 and "hemo_mu" in out.columns:
        hemo_mu = out["hemo_mu"].to_numpy(dtype=np.float64)
        # high hemo_mu = safe, factor → 1; low = unsafe, factor → 0.
        hemo_safety = _sigmoid_temp(hemo_mu - hemo_safe_log10, temp)
        # use (1 − hemo_penalty) + hemo_penalty * safety: hemo_penalty=0 → factor=1, =1 → depends entirely on safety
        hemo_factor = (1.0 - hemo_penalty) + hemo_penalty * hemo_safety
    else:
        hemo_factor = np.ones(len(out))

    narrow = amp_cal * target_score * sel_term * hemo_factor

    if strict:
        gate = (amp_cal >= thr_amp) & (target_score >= thr_target)
        narrow = np.where(gate, narrow, 0.0)

    out["target_score"] = target_score
    out["offtarget_score"] = offtarget_score
    out["pmic_T_bar"] = pmic_T_bar
    out["pmic_O_bar"] = pmic_O_bar
    out["selectivity"] = selectivity
    out["hemo_factor"] = hemo_factor
    out["narrow_score"] = narrow

    return out.sort_values("narrow_score", ascending=False, kind="mergesort").reset_index(drop=True)


def diversity_rerank(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    lambda_div: float = 0.1,
    top_k: int = 200,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """For the top_k candidates already ranked by NarrowScore, apply a diversity discount
    based on embedding cosine neighbors and rerank.

    DiverseScore_i = NarrowScore_i − λ_div · mean(top-n_neighbors cosine(emb_i, emb_j)),
    where j has already been selected into the ranked list before i.
    """
    df = df.copy().reset_index(drop=True)
    n = min(top_k, len(df))
    if n == 0 or embeddings is None or len(embeddings) != len(df):
        return df

    emb = embeddings[:n].astype(np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-6)
    emb = emb / norm

    selected: list[int] = [0]   # rank 0 is taken directly
    diverse_score = np.full(n, np.nan, dtype=np.float64)
    diverse_score[0] = float(df.loc[0, "narrow_score"])

    for i in range(1, n):
        sims = emb[i] @ emb[selected].T
        topn = np.sort(sims)[-n_neighbors:].mean() if sims.size else 0.0
        diverse_score[i] = float(df.loc[i, "narrow_score"] - lambda_div * topn)
        selected.append(i)

    df.loc[:n - 1, "diverse_score"] = diverse_score
    return df.sort_values(["diverse_score", "narrow_score"],
                           ascending=[False, False], kind="mergesort").reset_index(drop=True)
