"""Stage 2 trainer for the compact (benchmark) model family.

Mirrors amp_prioritizer.train's structure but:
  * model: AMPStrippedModel (no physchem/hemolysis heads)
  * loss:  stripped_loss (no L_phys, L_hemo, L_selectivity, L_consistency)
  * data:  AMPTrainWithNegatives (positives + AMPlify negatives 1:1)
  * eval:  only AMP/sp/MIC metrics computed, hemo/physchem skipped

Composite metric used for "best.pt" selection:
    score = AUROC + macro_f1 + mic/macro_spearman + 0.5 * mic/macro_ndcg_50

Launch (8 GPU full-FT):
    accelerate launch --num_processes 8 --mixed_precision fp16 \
        -m amp_prioritizer.train_stripped --config configs/benchmark.yaml

Single-GPU smoke test:
    python -m amp_prioritizer.train_stripped --config configs/benchmark.yaml --max_steps 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

import numpy as np
import pandas as pd
import torch
import torch.distributed.nn.functional as dist_nn
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from scipy.special import log_ndtr, ndtr
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, f1_score,
                             mean_absolute_error, ndcg_score, roc_auc_score)
from torch.utils.data import DataLoader

from amp_prioritizer.schema import load_latest_species

from amp_prioritizer.train_data import (
    AMPCollator,
    SpeciesGroupedBatchSampler,
    make_train_val_test,
    species_frequency_weights,
)
from amp_prioritizer.losses_stripped import stripped_loss
from amp_prioritizer.model_stripped import AMPStrippedModel


def gather_rank_tensor(
    tensor: torch.Tensor, accelerator: Accelerator, *, with_grad: bool,
) -> torch.Tensor:
    """Gather equal local rank pools across DDP processes."""
    if accelerator.num_processes == 1:
        return tensor
    if with_grad:
        return torch.cat(dist_nn.all_gather(tensor), dim=0)
    return accelerator.gather(tensor)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", default="data/current")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--finetune_mode", choices=["full", "lora"], default=None)
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, species_list, train_species_counts=None):
    model.eval()
    amp_logits, amp_targets = [], []
    sp_logits, sp_targets = [], []
    mic_mus, mic_distribution_mus, mic_log_sigmas = [], [], []
    mic_targets, mic_masks, mic_cr, mic_cl = [], [], [], []

    for batch in loader:
        out = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda())
        amp_logits.append(out["amp_logit"].cpu())
        sp_logits.append(out["species_logit"].cpu())
        mic_distribution_mus.append(out["mic_mu"].cpu())
        mic_mus.append(out.get("mic_point", out["mic_mu"]).cpu())
        mic_log_sigmas.append(out["mic_log_sigma"].cpu())
        amp_targets.append(batch["amp_label"].cpu())
        sp_targets.append(batch["species_label"].cpu())
        mic_targets.append(batch["mic_target"].cpu())
        mic_masks.append(batch["mic_mask"].cpu())
        mic_cr.append(batch["mic_censored_right"].cpu())
        mic_cl.append(batch["mic_censored_left"].cpu())

    amp_logits = torch.cat(amp_logits).float().numpy()
    sp_logits = torch.cat(sp_logits).float().numpy()
    mic_mus = torch.cat(mic_mus).float().numpy()
    mic_distribution_mus = torch.cat(mic_distribution_mus).float().numpy()
    mic_log_sigmas = torch.cat(mic_log_sigmas).float().numpy()
    amp_targets = torch.cat(amp_targets).numpy()
    sp_targets = torch.cat(sp_targets).numpy()
    mic_targets = torch.cat(mic_targets).numpy()
    mic_masks = torch.cat(mic_masks).numpy().astype(bool)
    mic_cr = torch.cat(mic_cr).numpy().astype(bool)
    mic_cl = torch.cat(mic_cl).numpy().astype(bool)

    metrics: dict[str, float] = {}
    m = amp_targets != -1
    if m.sum() > 0 and len(np.unique(amp_targets[m])) > 1:
        metrics["amp/auroc"] = float(roc_auc_score(amp_targets[m], amp_logits[m]))
        metrics["amp/auprc"] = float(average_precision_score(amp_targets[m], amp_logits[m]))
        metrics["amp/f1"] = float(f1_score(amp_targets[m], (amp_logits[m] > 0).astype(int)))

    f1s = []
    for i, sp in enumerate(species_list):
        ms = sp_targets[:, i] != -1
        if ms.sum() < 10 or len(np.unique(sp_targets[ms, i])) < 2:
            continue
        f1s.append(float(f1_score(sp_targets[ms, i], (sp_logits[ms, i] > 0).astype(int))))
    if f1s:
        metrics["sp/macro_f1"] = sum(f1s) / len(f1s)

    active = mic_masks & ~mic_cr & ~mic_cl
    rhos, ndcgs, maes = [], [], []
    for i, sp in enumerate(species_list):
        ms = active[:, i]
        if ms.sum() < 20:
            continue
        y_true = mic_targets[ms, i]
        y_pred = mic_mus[ms, i]
        if np.std(y_true) < 1e-6:
            continue
        try:
            rho = float(spearmanr(y_true, y_pred)[0])
            if not np.isnan(rho):
                metrics[f"mic/{sp}/spearman"] = rho
                rhos.append(rho)
        except Exception:
            pass
        mae = float(mean_absolute_error(y_true, y_pred))
        metrics[f"mic/{sp}/mae_log"] = mae
        maes.append(mae)
        k = int(min(50, len(y_true)))
        try:
            rel = y_true.max() - y_true
            ndcg = float(ndcg_score(rel[None, :], (-y_pred)[None, :], k=k))
            metrics[f"mic/{sp}/ndcg_50"] = ndcg
            ndcgs.append(ndcg)
        except Exception:
            pass
    if rhos:
        metrics["mic/macro_spearman"] = sum(rhos) / len(rhos)
    if maes:
        metrics["mic/macro_mae_log"] = sum(maes) / len(maes)
    if ndcgs:
        metrics["mic/macro_ndcg_50"] = sum(ndcgs) / len(ndcgs)
    if train_species_counts is not None:
        counts = np.asarray(train_species_counts, dtype=float)
        positive = counts[counts > 0]
        if len(positive):
            cutoff = float(np.quantile(positive, 1.0 / 3.0))
            low_rhos = [
                metrics[f"mic/{sp}/spearman"]
                for i, sp in enumerate(species_list)
                if counts[i] <= cutoff and f"mic/{sp}/spearman" in metrics
            ]
            if low_rhos:
                metrics["mic/low_resource_macro_spearman"] = float(np.mean(low_rhos))
                metrics["mic/low_resource_train_cutoff"] = cutoff

    censored = mic_masks & (mic_cr | mic_cl)
    if censored.any():
        log_sig = np.clip(mic_log_sigmas[censored], -2.0, 2.0)
        sig = np.exp(log_sig)
        z = (mic_targets[censored] - mic_distribution_mus[censored]) / sig
        right = mic_cr[censored]
        log_probability = np.where(right, log_ndtr(-z), log_ndtr(z))
        probability = np.exp(log_probability)
        violation = np.where(
            right, mic_mus[censored] < mic_targets[censored],
            mic_mus[censored] > mic_targets[censored],
        )
        metrics["mic/censored_nll"] = float(-np.mean(log_probability))
        metrics["mic/constraint_probability"] = float(np.mean(probability))
        metrics["mic/interval_violation_rate"] = float(np.mean(violation))
        metrics["mic/censored_count"] = int(censored.sum())
        metrics["mic/censoring_sanity_frac"] = float(1.0 - np.mean(violation))

    if active.any():
        exact_sigma = np.exp(np.clip(mic_log_sigmas[active], -2.0, 2.0))
        exact_z = np.abs((mic_targets[active] - mic_distribution_mus[active]) / exact_sigma)
        metrics["mic/exact_interval_coverage_50"] = float(np.mean(exact_z <= 0.67448975))
        metrics["mic/exact_interval_coverage_80"] = float(np.mean(exact_z <= 1.28155157))
        metrics["mic/exact_interval_coverage_95"] = float(np.mean(exact_z <= 1.95996398))
        metrics["mic/exact_standardized_residual_rmse"] = float(
            np.sqrt(np.mean(exact_z ** 2))
        )
    return metrics


def selection_key(
    metrics: dict, mode: str = "spearman_first",
    guardrails: dict | None = None,
) -> tuple[float, ...]:
    """Validation-only checkpoint ordering shared by paired ablation arms."""
    spearman = metrics.get("mic/macro_spearman", float("-inf"))
    low_resource = metrics.get("mic/low_resource_macro_spearman", float("-inf"))
    ndcg = metrics.get("mic/macro_ndcg_50", float("-inf"))
    neg_mae = -metrics.get("mic/macro_mae_log", float("inf"))
    if mode == "spearman_first":
        return spearman, low_resource, ndcg, neg_mae
    if mode == "ndcg_first":
        return ndcg, spearman, neg_mae, low_resource
    if mode == "point_guardrailed":
        rules = guardrails or {}
        checks = (
            ("mic/censored_nll", "max_censored_nll", lambda value, limit: value <= limit),
            ("mic/constraint_probability", "min_constraint_probability", lambda value, limit: value >= limit),
            ("mic/macro_mae_log", "max_macro_mae", lambda value, limit: value <= limit),
            ("amp/auroc", "min_amp_auroc", lambda value, limit: value >= limit),
            ("amp/auprc", "min_amp_auprc", lambda value, limit: value >= limit),
            ("amp/f1", "min_amp_f1", lambda value, limit: value >= limit),
            ("hemo/mae_log", "max_hemo_mae", lambda value, limit: value <= limit),
        )
        eligible = all(
            key not in rules
            or (metric in metrics and check(float(metrics[metric]), float(rules[key])))
            for metric, key, check in checks
        )
        neg_censored_nll = -metrics.get("mic/censored_nll", float("inf"))
        return float(eligible), spearman, ndcg, neg_mae, neg_censored_nll
    raise ValueError(f"unsupported checkpoint_selection: {mode!r}")


def apply_rank_schedule(
    loss_kwargs: dict, progress: float, schedule: dict | None,
) -> dict:
    """Scale ranking terms after pointwise predictions have stabilized."""
    if not schedule or not schedule.get("enabled", False):
        return loss_kwargs
    start = float(schedule.get("start_fraction", 0.4))
    ramp = float(schedule.get("ramp_fraction", 0.2))
    if not 0.0 <= start <= 1.0 or ramp < 0.0:
        raise ValueError(
            "rank_schedule requires 0 <= start_fraction <= 1 and ramp_fraction >= 0"
        )
    scale = (
        0.0 if progress < start
        else min(1.0, (progress - start) / max(ramp, 1e-12))
    )
    for key in (
        "alpha_lambda", "alpha_listmle", "alpha_rank_exact",
        "alpha_rank_censored", "alpha_rank_pooled",
    ):
        if key in loss_kwargs:
            loss_kwargs[key] *= scale
    return loss_kwargs


def config_sha256(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    accelerator = Accelerator(
        mixed_precision=cfg.get("mixed_precision", "fp16"),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False),
            InitProcessGroupKwargs(timeout=timedelta(hours=2)),
        ],
    )
    set_seed(cfg.get("seed", 42))
    if accelerator.is_main_process:
        print("== benchmark run config ==")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))

    species_list = load_latest_species(args.data_dir)
    n_species = len(species_list)

    mcfg = cfg["model"]
    finetune_mode = args.finetune_mode or mcfg.get("finetune_mode", "full")
    species_encoder_cfg = mcfg.get("species_encoder") or {"type": "embedding"}
    model = AMPStrippedModel(
        backbone_name=mcfg["name"],
        n_species=n_species,
        finetune_mode=finetune_mode,
        head_hidden=mcfg.get("head_hidden", 512),
        head_dropout=mcfg.get("head_dropout", 0.2),
        gradient_checkpointing=mcfg.get("gradient_checkpointing", True),
        hf_cache_dir=cfg.get("hf_cache_dir"),
        species_encoder_cfg=species_encoder_cfg,
        species_list=species_list,
        species_fusion=mcfg.get("species_fusion", "concat"),
        mic_head_mode=mcfg.get("mic_head_mode", "distribution"),
    )
    initial_checkpoint = cfg["train"].get("initial_checkpoint")
    if initial_checkpoint:
        checkpoint_path = ROOT / initial_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"initial checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
    if accelerator.is_main_process:
        tr, tot = model.trainable_param_count()
        print(f"trainable / total: {tr/1e6:.1f}M / {tot/1e6:.1f}M ({tr/tot:.1%})  "
              f"mode={finetune_mode}  n_species={n_species}")

    exclude_path = cfg["data"].get("exclude_dedup_keys_path")
    exclude_keys = None
    if exclude_path:
        exclude_table = pd.read_csv(ROOT / exclude_path)
        exclude_keys = set(exclude_table["dedup_key"].astype(str))
    ds = make_train_val_test(
        data_dir=args.data_dir,
        max_len=cfg["data"].get("max_len", 200),
        physchem_cache_dir=cfg["data"].get("physchem_cache_dir",
                                           f"{args.data_dir}/physchem"),
        active_threshold_log10=cfg["data"].get("active_threshold_log10", 1.5),
        neg_fasta=str(ROOT / cfg["data"]["neg_fasta"]),
        neg_log10_mic=cfg["data"].get("neg_log10_mic", 3.91),
        exclude_dedup_keys=exclude_keys,
    )
    if accelerator.is_main_process:
        print(f"train n={len(ds['train'])}  val n={len(ds['val'])}  test n={len(ds['test'])}")

    collator = AMPCollator(model.tokenizer, max_len=cfg["data"].get("max_len", 200) + 2)
    bsz = args.batch_size or cfg["train"]["per_device_batch_size"]
    grouped_cfg = cfg["train"].get("grouped_batch", {})
    if grouped_cfg.get("enabled", False):
        batch_sampler = SpeciesGroupedBatchSampler(
            ds["train"], bsz,
            species_per_batch=grouped_cfg.get("species_per_batch", 2),
            samples_per_species=grouped_cfg.get("samples_per_species", 2),
            negative_fraction=grouped_cfg.get("negative_fraction", 0.5),
            seed=cfg.get("seed", 42),
            drop_last=True,
        )
        train_loader = DataLoader(
            ds["train"], batch_sampler=batch_sampler,
            num_workers=cfg["train"].get("num_workers", 4),
            collate_fn=collator, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            ds["train"], batch_size=bsz, shuffle=True,
            num_workers=cfg["train"].get("num_workers", 4),
            collate_fn=collator, pin_memory=True, drop_last=True,
        )
    val_loader = DataLoader(ds["val"], batch_size=bsz * 2, shuffle=False,
                            num_workers=2, collate_fn=collator, pin_memory=True)
    exact_counts = (
        ds["train"].mic_mask & ~ds["train"].mic_cr & ~ds["train"].mic_cl
    ).sum(axis=0)
    frequency_weights = torch.from_numpy(species_frequency_weights(ds["train"]))

    head_keywords = ["amp_head", "species_head", "species_emb",
                     "species_encoder", "species_fusion", "species_mic_head",
                     "mic_global_head", "mic_residual_head"]
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if any(k in n for k in head_keywords) else backbone_params).append(p)
    lr_backbone = cfg["train"].get("lr_backbone", 2.0e-5)
    lr_head = cfg["train"].get("lr_head", 1.0e-4)
    wd = cfg["train"].get("wd", 0.01)
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": wd},
        {"params": head_params, "lr": lr_head, "weight_decay": wd},
    ])
    model, optim, train_loader = accelerator.prepare(
        model, optim, train_loader)
    updates_per_epoch = math.ceil(
        len(train_loader) / accelerator.gradient_accumulation_steps
    )
    total_steps = args.max_steps or updates_per_epoch * cfg["train"]["epochs"]
    warmup = int(total_steps * cfg["train"].get("warmup_ratio", 0.05))

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        progress = min(1.0, (s - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    out_dir = Path(cfg.get("out_dir", "runs/benchmark"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process and species_encoder_cfg.get("type") == "kmer_mlp":
        import shutil as _sh
        src_npz = species_encoder_cfg.get("kmer_features_path")
        if src_npz and Path(src_npz).exists():
            _sh.copyfile(src_npz, out_dir / "species_kmer_features.npz")

    base_loss_kwargs = dict(
        alpha_amp=cfg["train"].get("alpha_amp", 1.0),
        alpha_species=cfg["train"].get("alpha_species", 0.5),
        alpha_censored=cfg["train"].get("alpha_censored", 0.3),
        censor_event_multiplier=cfg["train"].get("censor_event_multiplier", 1.0),
        alpha_huber_active=cfg["train"].get("alpha_huber_active", 0.2),
        alpha_lambda=cfg["train"].get("alpha_lambda", 0.3),
        alpha_listmle=cfg["train"].get("alpha_listmle", 0.2),
        alpha_rank_exact=cfg["train"].get("alpha_rank_exact", 0.0),
        alpha_rank_censored=cfg["train"].get("alpha_rank_censored", 0.0),
        alpha_rank_pooled=cfg["train"].get("alpha_rank_pooled", 0.0),
        label_smoothing=cfg["train"].get("label_smoothing", 0.1),
        huber_delta=cfg["train"].get("huber_delta", 1.0),
        species_reduction=cfg["train"].get("species_reduction", "cell_mean"),
        species_weights=frequency_weights,
        censoring_mode=cfg["train"].get("censoring_mode", "proper"),
        alpha_point_consistency=cfg["train"].get("alpha_point_consistency", 0.0),
        rank_loss_type=cfg["train"].get("rank_loss_type", "gap_pairwise"),
        rank_min_gap_log10=cfg["train"].get("rank_min_gap_log10", 0.0),
        rank_ndcg_k=cfg["train"].get("rank_ndcg_k", 50),
        listmle_top_weighted=cfg["train"].get("listmle_top_weighted", False),
    )
    curriculum = cfg["train"].get("curriculum", {})
    rank_schedule = cfg["train"].get("rank_schedule", {})
    checkpoint_selection = cfg["train"].get("checkpoint_selection", "spearman_first")
    checkpoint_guardrails = cfg["train"].get("checkpoint_guardrails", {})

    best_key = (float("-inf"),)
    best_metrics: dict[str, float] = {}
    step = 0
    log_every = cfg["train"].get("log_every", 20)
    eval_every = cfg["train"].get("eval_every", 500)
    max_steps = args.max_steps or total_steps
    rank_batches = 0
    rank_nonzero_batches = 0
    rank_pair_totals = {"exact": 0, "censored": 0, "pooled": 0}
    t0 = time.time()
    while step < max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(batch["input_ids"], batch["attention_mask"])
                loss_kwargs = dict(base_loss_kwargs)
                if curriculum.get("enabled", False):
                    switch_fraction = float(curriculum.get("switch_fraction", 0.4))
                    if step / max(1, max_steps) >= switch_fraction:
                        loss_kwargs.update(curriculum.get("phase2", {}))
                loss_kwargs = apply_rank_schedule(
                    loss_kwargs, step / max(1, max_steps), rank_schedule)
                rank_kwargs = {}
                if (
                    cfg["train"].get("rank_global_pool", False)
                    and loss_kwargs.get("rank_loss_type") == "global_v2"
                ):
                    point = out.get("mic_point", out["mic_mu"])
                    rank_kwargs = {
                        "rank_pred": gather_rank_tensor(point, accelerator, with_grad=True),
                        "rank_target": gather_rank_tensor(batch["mic_target"], accelerator, with_grad=False),
                        "rank_mask": gather_rank_tensor(batch["mic_mask"], accelerator, with_grad=False),
                        "rank_censored_right": gather_rank_tensor(
                            batch["mic_censored_right"], accelerator, with_grad=False,
                        ),
                        "rank_censored_left": gather_rank_tensor(
                            batch["mic_censored_left"], accelerator, with_grad=False,
                        ),
                    }
                loss, parts = stripped_loss(
                    out["amp_logit"], out["species_logit"],
                    out["mic_mu"], out["mic_log_sigma"],
                    batch["amp_label"], batch["species_label"],
                    batch["mic_target"], batch["mic_mask"],
                    batch["mic_censored_right"],
                    mic_censored_left=batch["mic_censored_left"],
                    mic_point=out.get("mic_point"),
                    **rank_kwargs, **loss_kwargs,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(),
                                                cfg["train"].get("grad_clip", 1.0))
                optim.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                sched.step()
                step += 1
                if sched.last_epoch != step:
                    raise RuntimeError(
                        f"scheduler/global-step mismatch: {sched.last_epoch} != {step}"
                    )
                if loss_kwargs.get("rank_loss_type") == "global_v2":
                    rank_batches += 1
                    rank_nonzero_batches += int(float(parts["rank_nonzero"]) > 0)
                    for name in rank_pair_totals:
                        rank_pair_totals[name] += int(float(parts[f"rank_{name}_pairs"]))
                if step % log_every == 0 and accelerator.is_main_process:
                    dt = time.time() - t0
                    parts_str = " ".join(
                        f"{k[5:]}={float(v):.3f}" for k, v in parts.items()
                        if k.startswith("loss_")
                    )
                    rank_str = (
                        f" pairs={int(float(parts['rank_exact_pairs']))}/"
                        f"{int(float(parts['rank_censored_pairs']))}/"
                        f"{int(float(parts['rank_pooled_pairs']))}"
                        if loss_kwargs.get("rank_loss_type") == "global_v2" else ""
                    )
                    print(f"[s2-strip {step:6d}/{max_steps}] "
                          f"loss={float(parts['loss']):.3f} {parts_str}{rank_str} "
                          f"lr={sched.get_last_lr()[0]:.2e} {step/dt:.2f}step/s",
                          flush=True)
                if step % eval_every == 0 or step >= max_steps:
                    if accelerator.is_main_process:
                        try:
                            unwrapped = accelerator.unwrap_model(model)
                            unwrapped.cuda()
                            metrics = evaluate(
                                unwrapped, val_loader, species_list,
                                train_species_counts=exact_counts,
                            )
                            print(f"[eval @ {step}] "
                                  + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                                  flush=True)
                            key = selection_key(metrics, checkpoint_selection, checkpoint_guardrails)
                            if key > best_key:
                                best_key = key
                                best_metrics = dict(metrics)
                                torch.save({
                                    "model": unwrapped.state_dict(),
                                    "metrics": metrics, "step": step,
                                    "n_species": n_species,
                                    "species_list": species_list,
                                    "finetune_mode": finetune_mode,
                                    "species_encoder_cfg": species_encoder_cfg,
                                    "backbone_name": mcfg["name"],
                                    "species_fusion": mcfg.get("species_fusion", "concat"),
                                    "mic_head_mode": mcfg.get("mic_head_mode", "distribution"),
                                }, out_dir / "best.pt")
                                print(f"  saved best ({key[0]:.4f}) -> {out_dir/'best.pt'}",
                                      flush=True)
                        except Exception as e:
                            print(f"[eval error] {e}", flush=True)
                            raise
                    accelerator.wait_for_everyone()
                if step >= max_steps:
                    break
        if step >= max_steps:
            break

    if accelerator.is_main_process:
        if cfg["train"].get("save_last", True):
            torch.save({
                "model": accelerator.unwrap_model(model).state_dict(),
                "n_species": n_species,
                "species_list": species_list,
                "finetune_mode": finetune_mode,
                "species_encoder_cfg": species_encoder_cfg,
                "backbone_name": mcfg["name"],
                "species_fusion": mcfg.get("species_fusion", "concat"),
                "mic_head_mode": mcfg.get("mic_head_mode", "distribution"),
            }, out_dir / "last.pt")
        metadata = {
            "status": "complete",
            "config_sha256": config_sha256(cfg),
            "seed": cfg.get("seed", 42),
            "best_selection_key": list(best_key),
            "best_guardrail_eligible": (
                bool(best_key[0]) if checkpoint_selection == "point_guardrailed" else None
            ),
            "best_metrics": best_metrics,
            "epochs": int(cfg["train"]["epochs"]),
            "train_examples": int(len(ds["train"])),
            "world_size": int(accelerator.num_processes),
            "ranking_diagnostics": {
                "batches": rank_batches,
                "nonzero_batches": rank_nonzero_batches,
                "nonzero_fraction": rank_nonzero_batches / max(1, rank_batches),
                "mean_pairs": {
                    key: value / max(1, rank_batches)
                    for key, value in rank_pair_totals.items()
                },
            },
            "updates_per_epoch": int(updates_per_epoch),
            "total_steps": int(total_steps),
            "global_steps": step,
            "scheduler_steps": sched.last_epoch,
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True)
        )
        print(f"done. best val macro Spearman = {best_metrics.get('mic/macro_spearman', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
