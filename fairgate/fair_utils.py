#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fair_utils.py

Reusable building blocks for fairness-aware ECAPA training/evaluation:
- GRL (gradient reversal)
- Causal gate (simple embedding-level gate)
- Loss terms: decorrelation, REx variance penalty, CSS (capacity/saturation) regularizers
- Multi-protocol eval helpers: EER/minDCF, fixed-FMR threshold, GARBE, group EER, etc.

This file is designed to be drop-in for the original TaoRuijie ECAPA-TDNN style codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextlib import contextmanager

@contextmanager
def temporarily_eval(model):
    """Temporarily switch a module to eval() and restore its original train/eval state.

    This is critical for evaluation because BatchNorm updates running stats even under no_grad().
    """
    was_training = getattr(model, 'training', False)
    model.eval()
    try:
        yield model
    finally:
        if was_training:
            model.train()

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------- GRL ----------------

class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradReverse.apply(x, self.lambd)


def warmup_scale(epoch: int, warmup_epochs: int, max_val: float = 1.0) -> float:
    if warmup_epochs <= 0:
        return float(max_val)
    # epoch is 1-indexed in most training loops
    t = min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))
    return float(max_val) * t


# ---------------- Fair config ----------------

@dataclass
class FairConfig:
    # Enable/disable whole fair branch (for baseline runs)
    enabled: bool = True

    # Gate / CSS
    css_target_ratio: float = 0.8           # expected confound ratio
    lambda_css_cap: float = 0.0             # capacity penalty weight
    lambda_css_sat: float = 0.0             # saturation penalty weight

    # Gender heads
    lambda_gender_s: float = 0.0            # supervised gender loss on confound branch
    lambda_gender_adv: float = 0.0          # adversarial gender loss on speaker branch
    grl_lambda: float = 1.0
    grl_warmup_epochs: int = 4
    adv_warmup_epochs: int = 0
    adv_hidden: int = 256
    adv_dropout: float = 0.0
    adv_noise_std: float = 0.02
    stopgrad_gender_to_emb: bool = True

    # Disentanglement
    lambda_decor: float = 0.0               # decorrelation between branches
    lambda_rex: float = 0.0                 # REx penalty on speaker loss across groups

    # Optional knob reserved for future (kept for compatibility with your CLI)
    lambda_fair_fmr: float = 0.0


# ---------------- Gate / heads ----------------
class CausalGate(nn.Module):
    def __init__(self, emb_dim: int, hidden: int = 256, init_ratio=None, init_const: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, emb_dim)

        if init_ratio is not None:
            r = float(init_ratio)
            r = min(max(r, 1e-4), 1.0 - 1e-4)
            b = math.log(r / (1.0 - r))
            if init_const:
                nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, b)

    def forward(self, e: torch.Tensor, strength: float = 1.0, logit_shift: float = 0.0) -> torch.Tensor:
        x = self.act(self.fc1(e))
        logits = self.fc2(x)  # [B, D]

        strength_t = torch.as_tensor(strength, device=logits.device, dtype=logits.dtype)
        shift_t = torch.as_tensor(logit_shift, device=logits.device, dtype=logits.dtype)

        # ---- make broadcasting explicit & safe ----
        # Accept:
        #   scalar
        #   [D]  -> [1, D]
        #   [B]  -> [B, 1]
        #   [B, D] -> keep
        B, D = logits.shape

        if strength_t.ndim == 1:
            if strength_t.numel() == D:
                strength_t = strength_t.view(1, D)
            elif strength_t.numel() == B:
                strength_t = strength_t.view(B, 1)

        if shift_t.ndim == 1:
            if shift_t.numel() == D:
                shift_t = shift_t.view(1, D)
            elif shift_t.numel() == B:
                shift_t = shift_t.view(B, 1)

        logits = logits * strength_t + shift_t
        return torch.sigmoid(logits)


class GenderHead(nn.Module):
    def __init__(self, emb_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, 2),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


# ---------------- Loss terms ----------------

def css_capacity_loss(g: torch.Tensor, target_ratio: float) -> torch.Tensor:
    # g: [B,D] in (0,1)
    return (g.mean() - float(target_ratio)) ** 2


def css_saturation_loss(g: torch.Tensor) -> torch.Tensor:
    # encourage gate values to be near 0 or 1
    return (g * (1.0 - g)).mean()


def decorrelation_loss(e_s: torch.Tensor, e_c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Barlow-Twins style cross-correlation penalty between two views (branches).
    """
    z1 = (e_s - e_s.mean(dim=0, keepdim=True)) / (e_s.std(dim=0, keepdim=True) + eps)
    z2 = (e_c - e_c.mean(dim=0, keepdim=True)) / (e_c.std(dim=0, keepdim=True) + eps)
    c = (z1.T @ z2) / float(z1.size(0))
    return (c ** 2).mean()


def rex_penalty(per_sample_loss: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    """
    Risk Extrapolation penalty: variance of mean losses across groups.

    per_sample_loss: [B] (speaker loss per sample)
    group: [B] int (e.g. gender: 0/1)
    """
    uniq = torch.unique(group)
    if uniq.numel() <= 1:
        return per_sample_loss.new_zeros(())
    means = []
    for g in uniq:
        mask = (group == g)
        if mask.any():
            means.append(per_sample_loss[mask].mean())
    if len(means) <= 1:
        return per_sample_loss.new_zeros(())
    means = torch.stack(means, dim=0)
    return means.var(unbiased=False)


# ---------------- Eval metrics (numpy) ----------------

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return (eer, threshold). EER is a fraction in [0,1]. labels: 1 target, 0 non-target."""
    s = scores.astype(np.float64)
    y = labels.astype(np.int32)
    thr = np.unique(s)
    if thr.size > 20000:
        idx = np.linspace(0, thr.size - 1, 20000).astype(int)
        thr = thr[idx]
    fnmr = np.array([(s[y == 1] < t).mean() if (y == 1).sum() else np.nan for t in thr])
    fmr  = np.array([(s[y == 0] >= t).mean() if (y == 0).sum() else np.nan for t in thr])
    i = np.nanargmin(np.abs(fmr - fnmr))
    eer = 0.5 * (fmr[i] + fnmr[i])
    return float(eer), float(thr[i])


def compute_minDCF(scores: np.ndarray, labels: np.ndarray, p_target: float = 0.01,
                   c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    s = scores.astype(np.float64)
    y = labels.astype(np.int32)
    thr = np.unique(s)
    if thr.size > 20000:
        idx = np.linspace(0, thr.size - 1, 20000).astype(int)
        thr = thr[idx]
    best = float("inf")
    for t in thr:
        fnmr = (s[y == 1] < t).mean() if (y == 1).sum() else np.nan
        fmr  = (s[y == 0] >= t).mean() if (y == 0).sum() else np.nan
        if np.isnan(fnmr) or np.isnan(fmr):
            continue
        d = c_miss * fnmr * p_target + c_fa * fmr * (1.0 - p_target)
        if d < best:
            best = d
    return float(best)


def pick_tau_at_fmr(scores: np.ndarray, labels: np.ndarray, fmr_target: float) -> float:
    imp = scores[labels.astype(np.int32) == 0]
    if imp.size == 0:
        return float("nan")
    q = max(0.0, min(1.0, 1.0 - float(fmr_target)))
    return float(np.quantile(imp, q))


def err_rates(scores: np.ndarray, labels: np.ndarray, tau: float) -> Tuple[float, float]:
    y = labels.astype(np.int32)
    s = scores.astype(np.float64)
    fmr  = (s[y == 0] >= tau).mean() if (y == 0).sum() else np.nan
    fnmr = (s[y == 1] <  tau).mean() if (y == 1).sum() else np.nan
    return float(fmr), float(fnmr)


def garbe(FPR_m: float, FNR_m: float, FPR_f: float, FNR_f: float, alpha: float = 0.5) -> float:
    if any(np.isnan([FPR_m, FNR_m, FPR_f, FNR_f])):
        return float("nan")
    return float(alpha * abs(FPR_m - FPR_f) + (1.0 - alpha) * abs(FNR_m - FNR_f))


def group_eer(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    e, _ = compute_eer(scores[mask], labels[mask])
    return float(e)

# ---------------------------------------------------------------------
# Public API aliases
# ---------------------------------------------------------------------
# These aliases provide clearer public names while preserving backward
# compatibility with the original experiment code.

def select_threshold_at_fmr(scores, labels, target_fmr):
    """Select a global threshold at the requested false match rate.

    Args:
        scores: Trial scores. Larger scores indicate more likely target trials.
        labels: Binary labels, where 1 is target and 0 is non-target.
        target_fmr: Desired false match rate, e.g., 0.01.

    Returns:
        A global score threshold.
    """
    return pick_tau_at_fmr(scores, labels, target_fmr)


def compute_error_rates_at_threshold(scores, labels, threshold):
    """Compute FMR and FNMR at a fixed global threshold.

    Args:
        scores: Trial scores. Larger scores indicate more likely target trials.
        labels: Binary labels, where 1 is target and 0 is non-target.
        threshold: Global decision threshold.

    Returns:
        fmr: False match rate on non-target trials.
        fnmr: False non-match rate on target trials.
    """
    return err_rates(scores, labels, threshold)


def compute_garbe_at_threshold(scores, labels, groups, threshold):
    """Compute GARBE at a fixed global threshold.

    This is a readable public alias for ``garbe``.
    """
    return garbe(scores, labels, groups, threshold)


def compute_group_eer(scores, labels, groups):
    """Compute group-wise EER.

    This is a readable public alias for ``group_eer``.
    """
    return group_eer(scores, labels, groups)
