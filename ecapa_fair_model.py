#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecapa_fair_model.py

A clean port of your fairness strategies onto the original ECAPA-TDNN code style:
- Keep the original speaker encoder (ECAPA_TDNN) intact.
- Add embedding-level causal gating and gender branches in the wrapper model.
- Provide a single entry point for both training and evaluation:
    - training_step(...)
    - extract_embedding(...)

Design goals:
- Minimal coupling to legacy "trainer wrapper" code patterns
- Robust for DDP (wrap the whole module with DDP)
- Baseline mode truly disables all fairness branches

This version adds eval-compatibility:
- extract_embedding accepts use_id/use_gate/extra kwargs to avoid TypeError during --eval
- extract_embedding uses self.speaker_encoder (not self.encoder) and auto-handles aug flag
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fair_utils import (
    FairConfig, CausalGate, GenderHead, GradientReversal, warmup_scale,
    css_capacity_loss, css_saturation_loss, decorrelation_loss, rex_penalty
)
from aamsoftmax import AAMSoftmax


class ECAPAFairModel(nn.Module):
    def __init__(
        self,
        speaker_encoder: nn.Module,
        n_class: int,
        emb_dim: int = 192,
        m: float = 0.2,
        s: float = 30.0,
        fair: Optional[FairConfig] = None,
        baseline: bool = False,
    ):
        super().__init__()
        self.speaker_encoder = speaker_encoder
        self.emb_dim = int(emb_dim)
        self.speaker_loss = AAMSoftmax(n_class=n_class, emb_dim=self.emb_dim, m=m, s=s)

        self.fair = fair if fair is not None else FairConfig(enabled=False)
        self.baseline = bool(baseline)

        # Build fair branches only if enabled and not baseline.
        self._fair_enabled = bool(self.fair.enabled) and (not self.baseline)

        if self._fair_enabled:
            self.gate = CausalGate(
                self.emb_dim,
                hidden=max(self.emb_dim, int(self.fair.adv_hidden)),
                init_ratio=float(self.fair.css_target_ratio),
                init_const=True,
            )
            self.gender_head_c = GenderHead(
                self.emb_dim, hidden=int(self.fair.adv_hidden), dropout=float(self.fair.adv_dropout)
            )
            self.gender_head_s = GenderHead(
                self.emb_dim, hidden=int(self.fair.adv_hidden), dropout=float(self.fair.adv_dropout)
            )
            self.grl = GradientReversal(lambd=float(self.fair.grl_lambda))
        else:
            self.gate = None
            self.gender_head_c = None
            self.gender_head_s = None
            self.grl = None

    def fair_enabled(self) -> bool:
        return self._fair_enabled

    def _encode(self, wav: torch.Tensor, aug: bool = False) -> torch.Tensor:
        """
        Unified encoder call.
        Many ECAPA implementations use forward(wav, aug=True/False), but some do not.
        """
        try:
            # fast path: if encoder supports 'aug' kwarg
            return self.speaker_encoder(wav, aug=aug)
        except TypeError:
            # fallback: encoder forward(wav)
            return self.speaker_encoder(wav)

    @torch.no_grad()
    def extract_embedding(
        self,
        wav: torch.Tensor,
        # ---- eval compatibility flags ----
        use_id: bool = True,
        use_gate: bool = True,
        gate_strength: float = 1.0,
        gate_logit_shift: float = 0.0,
        # ---- optional returns ----
        return_gate: bool = False,
        return_confound: bool = False,
        # ---- swallow any extra args from other versions to avoid TypeError ----
        **kwargs: Any,
    ):
        """
        Eval-only helper: extract normalized embedding.

        Compatibility notes:
        - Some eval pipelines call extract_embedding(..., use_id=..., use_gate=...)
        - We accept **kwargs to avoid breaking when the caller passes extra flags.
        """
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            emb = self._encode(wav, aug=False)
            emb = F.normalize(emb, p=2, dim=-1)

            gate = None
            if self.fair_enabled() and use_gate and (self.gate is not None):
                emb_s, emb_c, gate = self._split(
                    emb,
                    use_gate=True,
                    gate_strength=gate_strength,
                    gate_logit_shift=gate_logit_shift,
                )
                out = emb_s if bool(use_id) else emb_c
                conf = emb_c
            else:
                out = emb
                conf = emb  # for return_confound compatibility; does not affect eval if unused

            out = F.normalize(out, p=2, dim=-1)

        if was_training:
            self.train()

        if return_confound and return_gate:
            return out, conf, gate
        if return_confound:
            return out, conf
        if return_gate:
            return out, gate
        return out

    def _split(
        self,
        emb: torch.Tensor,
        use_gate: bool = True,
        gate_strength: float = 1.0,
        gate_logit_shift: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns (emb_s, emb_c, g)
        emb_s: speaker branch (id)
        emb_c: confound branch
        g: gate in (0,1)
        """
        if (not use_gate) or (self.gate is None):
            # Degenerate split (no gate): put everything into speaker branch
            g = emb.new_zeros(emb.shape)  # shape [B, D]
            emb_c = emb * 0.0
            emb_s = emb
            return emb_s, emb_c, g

        g = self.gate(emb, strength=gate_strength, logit_shift=gate_logit_shift)
        emb_c = emb * g
        emb_s = emb * (1.0 - g)
        return emb_s, emb_c, g

    def training_step(
        self,
        wav: torch.Tensor,               # [B,T]
        spk: torch.Tensor,               # [B]
        gender: Optional[torch.Tensor],  # [B] 0/1 (M/F), can be None for baseline
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns (loss, stats_dict) where stats contain individual terms for logging.
        """
        self.train()

        emb = self._encode(wav, aug=True)
        emb = F.normalize(emb, dim=-1)

        # speaker embedding used for ID loss
        if self.fair_enabled():
            emb_s, emb_c, g = self._split(emb)
        else:
            emb_s, emb_c, g = emb, None, None

        # speaker loss (per-sample needed for REx)
        need_per = self.fair_enabled() and (float(self.fair.lambda_rex) > 0.0) and (gender is not None)
        L_id, prec1, per_sample = self.speaker_loss(emb_s, spk, return_per_sample=need_per)

        loss = L_id
        stats: Dict[str, float] = {"L_id": float(L_id.detach().cpu()), "prec1": float(prec1)}

        if not self.fair_enabled():
            return loss, stats

        # Warmups
        grl_scale = warmup_scale(epoch, int(self.fair.grl_warmup_epochs), max_val=float(self.fair.grl_lambda))
        adv_scale = warmup_scale(epoch, int(self.fair.adv_warmup_epochs), max_val=1.0)

        # Update GRL lambda dynamically
        self.grl.lambd = float(grl_scale)

        # -------- gender supervised on confound branch (keep gender there)
        L_gs = emb_s.new_zeros(())
        if (gender is not None) and (float(self.fair.lambda_gender_s) > 0.0):
            e_for_gs = emb_c.detach() if bool(self.fair.stopgrad_gender_to_emb) else emb_c
            logits_c = self.gender_head_c(e_for_gs)
            L_gs = F.cross_entropy(logits_c, gender)
            loss = loss + float(self.fair.lambda_gender_s) * L_gs
        stats["L_gs"] = float(L_gs.detach().cpu())

        # -------- gender adversarial on speaker branch (remove gender from emb_s)
        L_adv = emb_s.new_zeros(())
        if (gender is not None) and (float(self.fair.lambda_gender_adv) > 0.0):
            e_adv = emb_s
            if float(self.fair.adv_noise_std) > 0.0:
                e_adv = e_adv + torch.randn_like(e_adv) * float(self.fair.adv_noise_std)
            e_adv = self.grl(e_adv)  # gradient reversal
            logits_s = self.gender_head_s(e_adv)
            L_adv = F.cross_entropy(logits_s, gender)
            loss = loss + (float(self.fair.lambda_gender_adv) * float(adv_scale)) * L_adv
        stats["L_adv"] = float(L_adv.detach().cpu())

        # -------- CSS regularizers on gate
        L_cap = emb_s.new_zeros(())
        if float(self.fair.lambda_css_cap) > 0.0:
            L_cap = css_capacity_loss(g, target_ratio=float(self.fair.css_target_ratio))
            loss = loss + float(self.fair.lambda_css_cap) * L_cap
        stats["L_cap"] = float(L_cap.detach().cpu())

        L_sat = emb_s.new_zeros(())
        if float(self.fair.lambda_css_sat) > 0.0:
            L_sat = css_saturation_loss(g)
            loss = loss + float(self.fair.lambda_css_sat) * L_sat
        stats["L_sat"] = float(L_sat.detach().cpu())

        # -------- decorrelation between branches
        L_decor = emb_s.new_zeros(())
        if float(self.fair.lambda_decor) > 0.0:
            L_decor = decorrelation_loss(emb_s, emb_c)
            loss = loss + float(self.fair.lambda_decor) * L_decor
        stats["L_decor"] = float(L_decor.detach().cpu())

        # -------- REx penalty across gender groups (variance of speaker loss)
        L_rex = emb_s.new_zeros(())
        if (gender is not None) and float(self.fair.lambda_rex) > 0.0 and (per_sample is not None):
            L_rex = rex_penalty(per_sample_loss=per_sample, group=gender)
            loss = loss + float(self.fair.lambda_rex) * L_rex
        stats["L_rex"] = float(L_rex.detach().cpu())

        # gate statistics (for debugging)
        stats["g_mean"] = float(g.mean().detach().cpu())
        stats["g_sat"] = float((g * (1.0 - g)).mean().detach().cpu())

        return loss, stats

    # ---------------- checkpoint IO ----------------

    def state_dict_with_meta(self) -> Dict[str, object]:
        return {
            "model": self.state_dict(),
            "fair": asdict(self.fair),
            "baseline": bool(self.baseline),
            "emb_dim": int(self.emb_dim),
        }

    def load_state_dict_safely(self, ckpt: Dict[str, object], strict: bool = False):
        sd = ckpt.get("model", ckpt)
        clean = {}
        for k, v in sd.items():
            kk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
            clean[kk] = v
        missing, unexpected = self.load_state_dict(clean, strict=strict)
        return list(missing), list(unexpected)
