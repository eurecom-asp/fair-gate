#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trainECAPAModel_fair_ddp.py

A robust DDP-capable training + evaluation script that ports your fairness strategies
onto the original ECAPA-TDNN code style.

Key properties:
- Works with single-GPU python execution for --eval (no dist init required).
- Works with torchrun for multi-GPU training (--ddp or WORLD_SIZE>1).
- Supports multi-protocol eval_list (comma-separated) and prints per-protocol + pooled.
- Adds causal gate + (gender supervised + gender adversarial) branches, plus decor/REx/CSS.

Assumptions:
- 16kHz audio input.
- Evaluation trial file format: "label enroll.wav test.wav" where label is 0/1.
- Gender map is a json dict: { "idxxxx": "m"/"f"/"male"/"female"/"0"/"1"/... }.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
import re
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torchaudio
import torchaudio.functional as AF

from fair_utils import FairConfig, compute_eer, compute_minDCF, pick_tau_at_fmr, err_rates, garbe
from ecapa_fair_model import ECAPAFairModel

# Import the original ECAPA encoder from your codebase.
# If your repo uses a different symbol name, change this import only.
from model import ECAPA_TDNN


# ---------------- utils ----------------

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def barrier():
    if is_dist():
        dist.barrier()


# rank0 logging (console + file; safe for torchrun)
_LOG_FH = None  # opened in main() on global rank0 only
_HIST_PATH = None  # save_path/train_history.tsv (global rank0 only)


def _env_rank() -> int:
    """Return global rank from env without requiring dist init."""
    try:
        return int(os.environ.get("RANK", "0"))
    except Exception:
        return 0


def master_print(*args, **kwargs):
    """Print on rank0 and (optionally) append the same line to train.log."""
    if get_rank() != 0:
        return
    print(*args, **kwargs, flush=True)
    global _LOG_FH
    if _LOG_FH is not None:
        msg = " ".join(str(x) for x in args)
        _LOG_FH.write(msg + "\\n")
        _LOG_FH.flush()


def parse_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def infer_spk_from_path(p: str) -> Optional[str]:
    m = re.search(r"(id\d{3,8})", p)
    return m.group(1) if m else None


def resolve_wav(path_str: str, root: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(Path(root) / p)


def parse_gender(val) -> int:
    if val is None:
        return -1
    s = str(val).strip().lower()
    if s in {"m", "male", "0"}:
        return 0
    if s in {"f", "female", "1"}:
        return 1
    try:
        return int(s)
    except Exception:
        return -1


def init_ddp(args):
    # torchrun sets env vars: RANK, WORLD_SIZE, LOCAL_RANK
    backend = str(args.dist_backend)
    dist.init_process_group(backend=backend, init_method="env://")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    barrier()


def cleanup_ddp():
    if is_dist():
        barrier()
        dist.destroy_process_group()


# ---------------- dataset ----------------

class VoxTrainDataset(Dataset):
    def __init__(
            self,
            list_file: str,
            wav_root: str,
            num_frames: int,
            gender_map: Optional[Dict[str, str]] = None,
            musan_path: Optional[str] = None,
            rir_path: Optional[str] = None,
    ):
        self.wav_root = str(wav_root)
        self.num_frames = int(num_frames)

        self.gender_map = gender_map if gender_map is not None else {}
        self.musan_path = musan_path
        self.rir_path = rir_path

        self.items: List[Tuple[str, str]] = []
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 1:
                    wav_rel = parts[0]
                    spk = infer_spk_from_path(wav_rel) or "unknown"
                else:
                    # support: "wav_path spk" or "spk wav_path" (heuristic)
                    a, b = parts[0], parts[1]
                    if ("/" in a) or (a.endswith(".wav")) or (a.endswith(".flac")):
                        wav_rel, spk = a, b
                    else:
                        spk, wav_rel = a, b
                self.items.append((wav_rel, spk))

        # build speaker id map
        spk_set = sorted({spk for _, spk in self.items})
        self.spk2id = {s: i for i, s in enumerate(spk_set)}

    def __len__(self):
        return len(self.items)

    def _load_wav(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(dim=0)  # mono
        if sr != 16000:
            wav = AF.resample(wav, sr, 16000)
        return wav

    def _random_crop(self, wav: torch.Tensor) -> torch.Tensor:
        max_audio = self.num_frames * 160 + 240
        if wav.numel() <= max_audio:
            # pad by wrapping
            shortage = max_audio - wav.numel()
            if shortage > 0:
                wav = torch.cat([wav, wav.repeat((shortage // wav.numel()) + 1)[:shortage]], dim=0)
            return wav
        start = random.randint(0, wav.numel() - max_audio)
        return wav[start:start + max_audio]

    def __getitem__(self, idx):
        wav_rel, spk = self.items[idx]
        wav_path = resolve_wav(wav_rel, self.wav_root)
        wav = self._load_wav(wav_path)
        wav = self._random_crop(wav)

        spk_id = self.spk2id[spk]

        g = -1
        # use inferred "idxxxx" if available; else try spk string
        spk_key = infer_spk_from_path(wav_rel) or spk
        if spk_key in self.gender_map:
            g = parse_gender(self.gender_map[spk_key])

        return wav, spk_id, g


def collate_train(batch):
    wavs, spk, g = zip(*batch)
    maxlen = max([w.numel() for w in wavs])
    out = []
    for w in wavs:
        if w.numel() < maxlen:
            out.append(torch.cat([w, w.new_zeros(maxlen - w.numel())], dim=0))
        else:
            out.append(w)
    wav = torch.stack(out, dim=0)
    spk = torch.tensor(spk, dtype=torch.long)
    g = torch.tensor(g, dtype=torch.long)
    return wav, spk, g


# ---------------- eval helpers ----------------

def read_trials(trial_file: str) -> List[Tuple[int, str, str]]:
    trials = []
    with open(trial_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            lab = int(parts[0])
            a = parts[1]
            b = parts[2]
            trials.append((lab, a, b))
    return trials

@torch.no_grad()
def embed_all(
    model: nn.Module,
    wav_list: List[str],
    batch_size: int,
    device: torch.device,
    gate_strength: float = 1.0,
    gate_logit_shift: float = 0.0,
    use_gate: bool = True,   # ✅ 新增：与 eval_one_protocol 的调用一致
) -> Dict[str, torch.Tensor]:
    model.eval()
    emb: Dict[str, torch.Tensor] = {}

    bs = int(batch_size)
    if bs <= 0:
        bs = 1

    for i in range(0, len(wav_list), bs):
        chunk = wav_list[i:i + bs]
        wavs = []
        for p in chunk:
            wav_path = str(p)
            wav_t, sr = torchaudio.load(wav_path)
            wav_t = wav_t.mean(dim=0)
            if sr != 16000:
                wav_t = AF.resample(wav_t, sr, 16000)
            wavs.append(wav_t)

        maxlen = max(w.numel() for w in wavs)
        padded = []
        for w in wavs:
            if w.numel() < maxlen:
                padded.append(torch.cat([w, w.new_zeros(maxlen - w.numel())], dim=0))
            else:
                padded.append(w)

        x = torch.stack(padded, dim=0).to(device)

        # ✅ 把 use_gate 传下去（你的 ECAPAFairModel.extract_embedding 支持 use_gate）:contentReference[oaicite:5]{index=5}
        e = model.extract_embedding(
            x,
            use_id=True,
            gate_strength=float(gate_strength),
            gate_logit_shift=float(gate_logit_shift),
            use_gate=bool(use_gate),
        )

        for p, v in zip(chunk, e.cpu()):
            emb[str(p)] = v

    return emb

def score_trials(emb: Dict[str, torch.Tensor], trials: List[Tuple[int, str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    scores = []
    labels = []
    for lab, a, b in trials:
        ea = emb[a]
        eb = emb[b]
        s = float(torch.dot(ea, eb))
        scores.append(s)
        labels.append(lab)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int32)


def eval_one_protocol(
        model: nn.Module,
        trial_file: str,
        eval_path_root: Optional[str],
        gender_map: Optional[Dict[str, str]],
        fixed_fmr: Optional[float],
        eval_batch_size: int,
        device: torch.device,
        args=None) -> Dict[str, float]:
    trials = read_trials(trial_file)

    # resolve paths if eval_path_root provided
    resolved = []
    for lab, a, b in trials:
        aa = resolve_wav(a, eval_path_root) if eval_path_root else a
        bb = resolve_wav(b, eval_path_root) if eval_path_root else b
        resolved.append((lab, aa, bb))

    uniq = sorted({w for _, a, b in resolved for w in (a, b)})
    # IMPORTANT: evaluation must run in eval() mode; otherwise BatchNorm/Dropout will contaminate running stats.
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        emb = embed_all(model, uniq, eval_batch_size, device, gate_strength=float(args.gate_strength_eval),
                        gate_logit_shift=float(args.gate_logit_shift_eval), use_gate=(not args.eval_disable_gate))
    if was_training:
        model.train()

    scores, labels = score_trials(emb, resolved)
    eer, tau_eer = compute_eer(scores, labels)
    mindcf = compute_minDCF(scores, labels, p_target=0.01)

    # fixed FMR threshold and GARBE
    tau_fmr = float("nan")
    garbe_eer = float("nan")
    garbe_fmr = float("nan")
    if fixed_fmr is not None:
        tau_fmr = pick_tau_at_fmr(scores, labels, float(fixed_fmr))
        # compute FPR/FNR at tau_fmr
        fnr, fpr = err_rates(scores, labels, tau_fmr)
        # GARBE components (uses gender_map)
        if gender_map:
            # -------- GARBE (based on enrollment speaker gender) --------
            enroll_gender = np.zeros(len(trials), dtype=np.int32) - 1
            for i, (_, a, _) in enumerate(trials):
                spk = infer_spk_from_path(a) or ""
                g = gender_map.get(spk, gender_map.get(str(spk), None))
                if g is None:
                    continue
                enroll_gender[i] = 0 if str(g).lower().startswith("m") else 1

            m_mask = (enroll_gender == 0)
            f_mask = (enroll_gender == 1)

            # GARBE@pooledEER (tau_eer)
            fpr_m, fnr_m = err_rates(scores[m_mask], labels[m_mask], tau_eer) if m_mask.any() else (np.nan, np.nan)
            fpr_f, fnr_f = err_rates(scores[f_mask], labels[f_mask], tau_eer) if f_mask.any() else (np.nan, np.nan)
            garbe_eer = garbe(fpr_m, fnr_m, fpr_f, fnr_f, alpha=0.5)

            # GARBE@fixedFMR (tau_fmr)
            garbe_fmr = float("nan")
            if not np.isnan(tau_fmr):
                fpr_m2, fnr_m2 = err_rates(scores[m_mask], labels[m_mask], tau_fmr) if m_mask.any() else (np.nan,
                                                                                                          np.nan)
                fpr_f2, fnr_f2 = err_rates(scores[f_mask], labels[f_mask], tau_fmr) if f_mask.any() else (np.nan,
                                                                                                          np.nan)
                garbe_fmr = garbe(fpr_m2, fnr_m2, fpr_f2, fnr_f2, alpha=0.5)
    # gender split FPR/FNR at tau_fmr (optional)
    fpr_m = fnr_m = fpr_f = fnr_f = float("nan")
    if fixed_fmr is not None and gender_map:
        # build per-trial gender from speaker ids inferred in path
        labs = []
        scores_m = []
        labels_m = []
        scores_f = []
        labels_f = []
        for (lab, a, b), sc in zip(resolved, scores):
            # infer id from enroll file path
            spk = infer_spk_from_path(a) or infer_spk_from_path(b)
            g = parse_gender(gender_map.get(spk, None)) if spk else -1
            if g == 0:
                scores_m.append(sc)
                labels_m.append(lab)
            elif g == 1:
                scores_f.append(sc)
                labels_f.append(lab)
        if len(scores_m) > 0:
            fnr_m, fpr_m = err_rates(np.asarray(scores_m), np.asarray(labels_m), tau_fmr)
        if len(scores_f) > 0:
            fnr_f, fpr_f = err_rates(np.asarray(scores_f), np.asarray(labels_f), tau_fmr)

    # overall FPR/FNR at tau_fmr
    fnr = fpr = float("nan")
    if fixed_fmr is not None:
        fnr, fpr = err_rates(scores, labels, tau_fmr)

    out = {
        "EER": float(eer * 100.0),
        "minDCF": float(mindcf),
        "tau_eer": float(tau_eer),
        "tau_fmr": float(tau_fmr),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "FPR_m": float(fpr_m),
        "FNR_m": float(fnr_m),
        "FPR_f": float(fpr_f),
        "FNR_f": float(fnr_f),
        "GARBE_eer": float(garbe_eer),
        "GARBE_fmr": float(garbe_fmr),
        "n_trials": float(len(trials)),
    }
    return out


# ---------------- checkpoint IO ----------------

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean = {}
    for k, v in sd.items():
        kk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        clean[kk] = v
    return clean


def load_ckpt(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
              strict: bool = False) -> int:
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    sd = _strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if get_rank() == 0:
        master_print(f"[CKPT] load={path} missing={len(missing)} unexpected={len(unexpected)}")
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def save_ckpt(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args):
    m = model.module if isinstance(model, DDP) else model
    ckpt = {
        "model": m.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_pretrained_ecapa(
        path: str,
        model: nn.Module,
        load_classifier: bool = True,
) -> Tuple[int, int]:
    """
    Load a vanilla ECAPA / ECAPAModel checkpoint into ECAPAFairModel.
    - Loads speaker_encoder always (if possible).
    - Loads speaker_loss (AAMSoftmax) only if load_classifier=True and shapes match.
    Returns: (num_loaded_keys, num_total_keys_in_ckpt)
    """
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    sd = _strip_module_prefix(sd)

    m = model.module if isinstance(model, DDP) else model
    model_sd = m.state_dict()
    model_keys = set(model_sd.keys())
    ckpt_keys = set(sd.keys())

    loaded = {}

    # 1) Direct overlap (if ckpt already has speaker_encoder.* / speaker_loss.*)
    overlap = model_keys & ckpt_keys
    for k in overlap:
        loaded[k] = sd[k]

    # 2) If no direct overlap, try map vanilla ECAPA keys -> wrapper keys
    if len(loaded) == 0:
        # common patterns:
        # - vanilla: "speaker_encoder.xxx" or "encoder.xxx" or "model.xxx"
        # - wrapper expects "speaker_encoder.xxx" and "speaker_loss.xxx"
        for k, v in sd.items():
            if k.startswith("speaker_encoder."):
                loaded[k] = v
            elif k.startswith("encoder."):
                loaded["speaker_encoder." + k[len("encoder."):]] = v
            elif k.startswith("model."):
                loaded_k = k[len("model."):]
                if loaded_k.startswith("speaker_encoder."):
                    loaded[loaded_k] = v

    # 3) Optionally load classifier if shapes match
    if not load_classifier:
        loaded = {k: v for k, v in loaded.items() if not k.startswith("speaker_loss.")}
    else:
        # keep only those with matching shapes
        cleaned = {}
        for k, v in loaded.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                cleaned[k] = v
        loaded = cleaned

    # apply partial load
    model_sd.update(loaded)
    m.load_state_dict(model_sd, strict=False)

    return len(loaded), len(sd)


# ---------------- main ----------------

def main():
    global _LOG_FH, _HIST_PATH
    parser = argparse.ArgumentParser()

    # core
    parser.add_argument("--train_list", type=str, required=False)
    parser.add_argument("--train_path", type=str, required=False)
    parser.add_argument("--eval_list", type=str, default="")
    parser.add_argument("--eval_path", type=str, default="")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_save_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=1024)

    # model/loss
    parser.add_argument("--n_class", type=int, required=True)
    parser.add_argument("--C", type=int, default=1024)  # channel width (original ECAPA)
    parser.add_argument("--m", type=float, default=0.2)
    parser.add_argument("--s", type=float, default=30.0)

    # training
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--test_step", type=int, default=5)
    parser.add_argument("--defer_eval", action="store_true")
    parser.add_argument("--amp", action="store_true")

    # ddp
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--dist_backend", type=str, default="nccl")

    # augmentation
    parser.add_argument("--musan_path", type=str, default="")
    parser.add_argument("--rir_path", type=str, default="")

    # gender map
    parser.add_argument("--gender_map", type=str, default="")
    parser.add_argument("--eval_gender_map", type=str, default="")

    # baseline: disable fair branches
    parser.add_argument("--baseline", action="store_true")

    # fairness hyper-params
    parser.add_argument("--css_target_ratio", type=float, default=0.03)
    parser.add_argument("--lambda_css_cap", type=float, default=0.0)
    parser.add_argument("--lambda_css_sat", type=float, default=0.0)
    parser.add_argument("--lambda_gender_s", type=float, default=0.0)
    parser.add_argument("--lambda_gender_adv", type=float, default=0.0)
    parser.add_argument("--lambda_decor", type=float, default=0.0)
    parser.add_argument("--lambda_rex", type=float, default=0.0)
    parser.add_argument("--lambda_fair_fmr", type=float, default=0.0)

    # adv head / GRL
    parser.add_argument("--grl_lambda", type=float, default=1.0)
    parser.add_argument("--grl_warmup_epochs", type=int, default=0)
    parser.add_argument("--adv_warmup_epochs", type=int, default=0)
    parser.add_argument("--adv_hidden", type=int, default=256)
    parser.add_argument("--adv_dropout", type=float, default=0.0)
    parser.add_argument("--adv_noise_std", type=float, default=0.0)
    parser.add_argument("--stopgrad_gender_to_emb", type=parse_bool, default=False)

    # eval
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=1)

    parser.add_argument("--gate_strength_eval", type=float, default=1.0,
                        help="Eval-time gate strength. Set 0 to make g a constant via logit shift.")
    parser.add_argument("--gate_logit_shift_eval", type=float, default=0.0,
                        help="Eval-time logit shift for gate. For constant g=r with strength=0, use log(r/(1-r)).")
    parser.add_argument("--fixed_fmr", type=float, default=None)
    parser.add_argument("--eval_disable_gate", action="store_true",
                        help="During evaluation, bypass the gate and use raw speaker embedding.")

    # resume
    parser.add_argument("--resume", type=str, default="")

    # finetune / pretrained
    parser.add_argument("--pretrained_ecapa", type=str, default="",
                        help="Path to vanilla ECAPA checkpoint to init speaker_encoder (and optionally classifier) before training. Used only if --resume is empty.")
    parser.add_argument("--pretrained_load_classifier", type=parse_bool, default=True,
                        help="Whether to also load the AAMSoftmax classifier from --pretrained_ecapa (only if shapes match).")
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0,
                        help="Freeze speaker_encoder parameters for the first N epochs (finetune).")
    parser.add_argument("--encoder_lr", type=float, default=0.0,
                        help="Optional separate LR for speaker_encoder (finetune). If 0, use --lr for all params.")

    args = parser.parse_args()
    seed_everything(args.seed)

    # model_save_path default
    if not args.model_save_path:
        args.model_save_path = str(Path(args.save_path) / "models")
    Path(args.model_save_path).mkdir(parents=True, exist_ok=True)

    # --- rank0 logging setup (avoid multiple ranks opening the same file before dist init)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if _env_rank() == 0:
        log_path = Path(args.save_path) / "train.log"
        _LOG_FH = open(log_path, "a", buffering=1, encoding="utf-8")
        _LOG_FH.write(f"\n\n===== RUN {datetime.now().isoformat()} =====\n")
        _LOG_FH.flush()

    # --- per-epoch TSV history (rank0 only; appended)
    _HIST_PATH = Path(args.save_path) / "train_history.tsv"
    if _env_rank() == 0 and (not _HIST_PATH.exists()):
        _HIST_PATH.write_text(
            "epoch\tlr\tloss\tL_id\tL_gs\tL_adv\tL_cap\tL_sat\tL_decor\tL_rex\tg_mean\tg_sat\tprec1\n",
            encoding="utf-8",
        )

    # Load gender maps
    train_gender_map = {}
    eval_gender_map = {}
    if args.gender_map and Path(args.gender_map).exists():
        train_gender_map = json.loads(Path(args.gender_map).read_text(encoding="utf-8"))
    if args.eval_gender_map and Path(args.eval_gender_map).exists():
        eval_gender_map = json.loads(Path(args.eval_gender_map).read_text(encoding="utf-8"))

    # Device and DDP
    if args.ddp and not args.eval:
        init_ddp(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_dist():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")

    # Build model
    encoder = ECAPA_TDNN(C=int(args.C))
    fair_cfg = FairConfig(
        enabled=not bool(args.baseline),
        css_target_ratio=float(args.css_target_ratio),
        lambda_css_cap=float(args.lambda_css_cap),
        lambda_css_sat=float(args.lambda_css_sat),
        lambda_gender_s=float(args.lambda_gender_s),
        lambda_gender_adv=float(args.lambda_gender_adv),
        lambda_decor=float(args.lambda_decor),
        lambda_rex=float(args.lambda_rex),
        lambda_fair_fmr=float(args.lambda_fair_fmr),
        grl_lambda=float(args.grl_lambda),
        grl_warmup_epochs=int(args.grl_warmup_epochs),
        adv_warmup_epochs=int(args.adv_warmup_epochs),
        adv_hidden=int(args.adv_hidden),
        adv_dropout=float(args.adv_dropout),
        adv_noise_std=float(args.adv_noise_std),
        stopgrad_gender_to_emb=bool(args.stopgrad_gender_to_emb),
    )

    model = ECAPAFairModel(
        speaker_encoder=encoder,
        n_class=int(args.n_class),
        emb_dim=192,
        m=float(args.m),
        s=float(args.s),
        fair=fair_cfg,
        baseline=bool(args.baseline),
    ).to(device)

    if is_dist():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)

    # -------- evaluation only
    if args.eval:
        if args.resume:
            load_ckpt(args.resume, model, optimizer=None, strict=False)
        elif Path(args.model_save_path).exists():
            # best-effort: load the latest checkpoint in model_save_path
            ckpts = sorted(Path(args.model_save_path).glob("model_*.pt"))
            if ckpts:
                load_ckpt(str(ckpts[-1]), model, optimizer=None, strict=False)

        eval_lists = [p.strip() for p in args.eval_list.split(",") if p.strip()]
        if not eval_lists:
            raise ValueError("--eval_list is empty.")
        master_print(f"[EVAL] protocols={len(eval_lists)} fixed_fmr={args.fixed_fmr}")

        # Per protocol
        merged_trials = []
        for p in eval_lists:
            metrics = eval_one_protocol(
                model=model.module if isinstance(model, DDP) else model,
                trial_file=p,
                eval_path_root=args.eval_path if args.eval_path else None,
                gender_map=eval_gender_map if eval_gender_map else None,
                fixed_fmr=float(args.fixed_fmr) if args.fixed_fmr is not None else None,
                eval_batch_size=int(args.eval_batch_size),
                device=device, args=args)
            master_print(
                f"[EVAL] {Path(p).name}: "
                f"EER={metrics['EER']:.4f}  minDCF={metrics['minDCF']:.4f}  "
                f"FPR={metrics['FPR']:.4f}  FNR={metrics['FNR']:.4f}  "
                f"FPR_m={metrics['FPR_m']:.4f}  FNR_m={metrics['FNR_m']:.4f}  "
                f"FPR_f={metrics['FPR_f']:.4f}  FNR_f={metrics['FNR_f']:.4f}  "
                f"GARBE_eer={metrics['GARBE_eer']:.4f}  GARBE_fmr={metrics['GARBE_fmr']:.4f}"
            )

            # For pooled
            trials = read_trials(p)
            resolved = []
            for lab, a, b in trials:
                aa = resolve_wav(a, args.eval_path) if args.eval_path else a
                bb = resolve_wav(b, args.eval_path) if args.eval_path else b
                resolved.append((lab, aa, bb))
            merged_trials.extend(resolved)

        # pooled
        if merged_trials:
            uniq = sorted({w for _, a, b in merged_trials for w in (a, b)})
            emb = embed_all(model.module if isinstance(model, DDP) else model, uniq, int(args.eval_batch_size), device)
            scores, labels = score_trials(emb, merged_trials)
            eer, tau = compute_eer(scores, labels)
            mindcf = compute_minDCF(scores, labels, p_target=0.01)
            master_print(f"[EVAL] POOLED: EER={eer * 100.0:.4f}  minDCF={mindcf:.6f}  n_trials={len(merged_trials)}")

        # --- close rank0 log file
        if get_rank() == 0 and _LOG_FH is not None:
            _LOG_FH.close()
            _LOG_FH = None

        return

    # -------- training
    if not args.train_list or not args.train_path:
        raise ValueError("--train_list and --train_path are required for training")

    ds = VoxTrainDataset(
        list_file=args.train_list,
        wav_root=args.train_path,
        num_frames=int(args.num_frames),
        gender_map=train_gender_map if train_gender_map else None,
        musan_path=args.musan_path if args.musan_path else None,
        rir_path=args.rir_path if args.rir_path else None,
    )

    sampler = None
    if is_dist():
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)

    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_train,
    )

    # Optimizer (support separate encoder_lr)
    if float(args.encoder_lr) > 0.0:
        m = model.module if isinstance(model, DDP) else model
        optimizer = torch.optim.Adam(
            [
                {"params": m.speaker_encoder.parameters(), "lr": float(args.encoder_lr)},
                {"params": [p for n, p in m.named_parameters() if not n.startswith("speaker_encoder.")],
                 "lr": float(args.lr)},
            ],
            weight_decay=2e-5,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=2e-5)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(args.lr_decay))

    # Pretrained init (only if not resuming and not eval)
    if (not args.resume) and args.pretrained_ecapa:
        n_loaded, n_total = load_pretrained_ecapa(
            args.pretrained_ecapa,
            model,
            load_classifier=bool(args.pretrained_load_classifier),
        )
        master_print(f"[PRETRAIN] loaded_keys={n_loaded} / ckpt_keys={n_total} from {args.pretrained_ecapa}")

    start_epoch = 1
    if args.resume:
        start_epoch = load_ckpt(args.resume, model, optimizer=optimizer, strict=False) + 1
        master_print(f"[RESUME] start_epoch={start_epoch}")

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))

    master_print(f"[TRAIN] world_size={get_world_size()}  steps/epoch={len(dl)}  save={args.save_path}")
    t0 = time.time()

    for epoch in range(start_epoch, int(args.max_epoch) + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)

        # optional freezing for finetune
        if int(args.freeze_encoder_epochs) > 0:
            m = model.module if isinstance(model, DDP) else model
            freeze = (epoch <= int(args.freeze_encoder_epochs))
            for p in m.speaker_encoder.parameters():
                p.requires_grad = (not freeze)
            master_print(f"[FINETUNE] epoch={epoch} encoder_frozen={freeze}")

        model.train()
        losses = []
        # stats accumulators (rank0 only)
        stat_sum: Dict[str, float] = {}
        stat_n = 0

        for it, (wav, spk, g) in enumerate(dl, start=1):
            wav = wav.to(device, non_blocking=True)
            spk = spk.to(device, non_blocking=True).long()
            g = g.to(device, non_blocking=True).long()
            # mask unknown gender -> None
            gender = g
            if (g < 0).any():
                # keep label but set weight by masking: simplest: set to 0 and ignore by excluding lambda losses
                # we handle by setting gender=None if any unknown in batch
                # (if you need partial masking, we can implement per-sample masking later)
                gender = None

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                m = model.module if isinstance(model, DDP) else model
                loss, stats = m.training_step(wav=wav, spk=spk, gender=gender, epoch=epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            if get_rank() == 0:
                for k, v in stats.items():
                    stat_sum[k] = stat_sum.get(k, 0.0) + float(v)
                stat_n += 1

            if get_rank() == 0 and (it % 20 == 0 or it == len(dl)):
                lrs = [pg["lr"] for pg in optimizer.param_groups]
                lr = lrs[0]
                msg = f"[E{epoch:03d}][{it:04d}/{len(dl)}] lr0={lr:.3e} lrs={[f'{x:.3e}' for x in lrs]} loss={np.mean(losses):.4f}"
                if stat_n > 0:
                    msg += " " + " ".join(
                        [f"{k}={stat_sum[k] / stat_n:.4f}" for k in sorted(stat_sum.keys()) if k != "prec1"])
                    msg += f" prec1={stat_sum.get('prec1', 0.0) / stat_n:.2f}"
                master_print(msg)

        # --- write per-epoch history row (rank0 only)
        if get_rank() == 0 and _HIST_PATH is not None and stat_n > 0:
            lr0 = float(optimizer.param_groups[0]["lr"])
            mean_loss = float(np.mean(losses)) if len(losses) else float("nan")

            def _avg(k: str) -> float:
                return float(stat_sum.get(k, float("nan"))) / float(stat_n) if stat_n > 0 else float("nan")

            row = [
                str(int(epoch)),
                f"{lr0:.6e}",
                f"{mean_loss:.6f}",
                f"{_avg('L_id'):.6f}",
                f"{_avg('L_gs'):.6f}",
                f"{_avg('L_adv'):.6f}",
                f"{_avg('L_cap'):.6f}",
                f"{_avg('L_sat'):.6f}",
                f"{_avg('L_decor'):.6f}",
                f"{_avg('L_rex'):.6f}",
                f"{_avg('g_mean'):.6f}",
                f"{_avg('g_sat'):.6f}",
                f"{_avg('prec1'):.6f}",
            ]
            with open(_HIST_PATH, "a", encoding="utf-8") as f:
                f.write("\t".join(row) + "\n")

        scheduler.step()

        # Save
        if get_rank() == 0 and (epoch % int(args.save_every) == 0 or epoch == int(args.max_epoch)):
            ckpt_path = str(Path(args.model_save_path) / f"model_{epoch:04d}.pt")
            save_ckpt(ckpt_path, model, optimizer, epoch, args)
            master_print(f"[SAVE] {ckpt_path}")

        # Eval during training (optional)
        if (not args.defer_eval) and args.eval_list and (epoch % int(args.test_step) == 0) and (get_rank() == 0):
            eval_lists = [p.strip() for p in args.eval_list.split(",") if p.strip()]
            master_print(f"[EVAL@E{epoch:03d}] protocols={len(eval_lists)}")
            for p in eval_lists:
                metrics = eval_one_protocol(
                    model=model.module if isinstance(model, DDP) else model,
                    trial_file=p,
                    eval_path_root=args.eval_path if args.eval_path else None,
                    gender_map=eval_gender_map if eval_gender_map else None,
                    fixed_fmr=float(args.fixed_fmr) if args.fixed_fmr is not None else None,
                    eval_batch_size=int(args.eval_batch_size),
                    device=device, args=args)
                master_print(
                    f"[EVAL@E{epoch:03d}] {Path(p).name}: "
                    + " ".join(
                        [f"{k}={v:.4f}" for k, v in metrics.items() if k not in {"tau_eer", "tau_fmr", "n_trials"}])
                )

        barrier()

    if get_rank() == 0:
        master_print(f"[DONE] total_time={(time.time() - t0) / 3600:.2f}h")

    # --- close rank0 log file
    if get_rank() == 0 and _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None

    cleanup_ddp()


if __name__ == "__main__":
    main()
