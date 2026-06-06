#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aamsoftmax.py

A minimal, reliable Additive Angular Margin Softmax (AAM-Softmax) implementation
compatible with the original ECAPA-TDNN training loop.

This avoids incidental coupling to legacy loss.py variations and makes the port cleaner.
"""

from __future__ import annotations
from typing import Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    def __init__(self, n_class: int, emb_dim: int = 192, m: float = 0.2, s: float = 30.0):
        super().__init__()
        self.m = float(m)
        self.s = float(s)
        self.weight = nn.Parameter(torch.randn(n_class, emb_dim))
        nn.init.xavier_normal_(self.weight, gain=1.0)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(
        self,
        emb: torch.Tensor,          # [B,D]
        label: torch.Tensor,        # [B]
        return_per_sample: bool = False,
    ) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
        """
        Return:
          loss (scalar), prec1 (float), per_sample_loss ([B] or None)
        """
        emb = F.normalize(emb, dim=1)
        W = F.normalize(self.weight, dim=1)

        # cosine: [B, n_class]
        cosine = F.linear(emb, W).clamp(-1.0, 1.0)
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Decision boundary correction
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.s

        ce = F.cross_entropy(logits, label, reduction='none' if return_per_sample else 'mean')
        if return_per_sample:
            loss = ce.mean()
            per_sample = ce
        else:
            loss = ce
            per_sample = None

        # Top-1 accuracy
        prec1 = (logits.argmax(dim=1) == label).float().mean().item() * 100.0
        return loss, prec1, per_sample
