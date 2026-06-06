#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ECAPAModel.py

- 训练：ECAPA-TDNN + AAM-Softmax
- 评估：支持返回 scores/labels；可选 AS-Norm（Adaptive S-Norm）
  - raw score 与原版口径一致：
      score_1 = mean( emb1_enr @ emb1_tst^T )   (full utterance, [1,D])
      score_2 = mean( emb2_enr @ emb2_tst^T )   (5 segments, [5,D] vs [5,D])
      score = 0.5*(score_1 + score_2)
    其中 score_2 等价于 dot(mean(emb2_enr), mean(emb2_tst))（均值线性可交换），
    所以实现 AS-Norm 时使用 v2 = mean(emb2, dim=0) 可保证数值一致且更快。

- EER/minDCF：默认使用 fair_utils（与 trainECAPAModel.py 的评估一致）
"""

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from loss import AAMsoftmax
from model import ECAPA_TDNN


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C, n_class, m, s, test_step, **kwargs):
        super().__init__()

        # ECAPA-TDNN encoder
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()

        # AAM-Softmax head
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).cuda()

        # Optimizer / scheduler
        params = list(self.speaker_encoder.parameters()) + list(self.speaker_loss.parameters())
        self.optim = torch.optim.Adam(params, lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=int(test_step), gamma=float(lr_decay))

        self.test_step = int(test_step)

    def train_network(self, epoch, loader):
        """Train one epoch. Returns (loss_avg, lr, acc_pct)."""
        self.train()
        self.scheduler.step()  # step per epoch (simple, stable)

        lr = self.optim.param_groups[0]['lr']
        index, top1, loss_sum = 0, 0.0, 0.0
        num = 0

        for num, (data, labels) in enumerate(loader, start=1):
            self.optim.zero_grad(set_to_none=True)

            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)

            nloss.backward()
            self.optim.step()

            bs = int(labels.size(0))
            index += bs
            top1 += float(prec)
            loss_sum += float(nloss.detach().cpu().item())

            sys.stderr.write(
                time.strftime("%m-%d %H:%M:%S") +
                f" [{epoch:2d}] Lr: {lr:.6f}, Training: {100.0 * (num / len(loader)):.2f}%, "
                f"Loss: {loss_sum / num:.5f}, ACC: {top1 / index * bs:.2f}% \r"
            )
            sys.stderr.flush()

        sys.stdout.write("\n")
        loss_avg = loss_sum / max(1, num)
        acc_pct = (top1 / index * bs) if index > 0 else 0.0
        return loss_avg, lr, acc_pct

    @torch.no_grad()
    def _embed_file(self, wav_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (embedding_1 [1,D], embedding_2 [5,D]) normalized."""
        audio, _ = soundfile.read(wav_path)

        # Full utterance
        data_1 = torch.FloatTensor(np.stack([audio], axis=0)).cuda()

        # Split utterance into 5 segments
        max_audio = 300 * 160 + 240
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')

        feats = []
        startframe = np.linspace(0, audio.shape[0] - max_audio, num=5)
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

        feats = np.stack(feats, axis=0).astype(np.float32, copy=False)
        data_2 = torch.FloatTensor(feats).cuda()

        embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
        embedding_1 = F.normalize(embedding_1, p=2, dim=1)  # [1,D]

        embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
        embedding_2 = F.normalize(embedding_2, p=2, dim=1)  # [5,D]

        return embedding_1, embedding_2

    @staticmethod
    def _read_list(path: str) -> List[str]:
        out = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(line)
        return out

    def eval_network(
        self,
        eval_list: str,
        eval_path: str,
        return_scores: bool = False,
        asnorm: bool = False,
        cohort_list: str = "",
        cohort_path: str = "",
        cohort_emb_path: str = "",
        asnorm_topk: int = 200,
        **kwargs
    ):
        """
        Returns:
          - if return_scores=False: (EER_pct, minDCF)
          - if return_scores=True : (EER_pct, minDCF, scores(list[float]), labels(list[int]))
        """
        self.eval()

        lines = self._read_list(eval_list)
        files = []
        for line in lines:
            ss = line.split()
            if len(ss) < 3:
                continue
            files.append(ss[1])
            files.append(ss[2])

        setfiles = sorted(set(files))

        # 1) compute eval embeddings
        embeddings: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for file in tqdm.tqdm(setfiles, desc="EvalEmb", total=len(setfiles)):
            wav_path = os.path.join(eval_path, file)
            embeddings[file] = self._embed_file(wav_path)

        # 2) AS-Norm: prepare cohort vectors and per-file (mu,std)
        mu: Dict[str, torch.Tensor] = {}
        sd: Dict[str, torch.Tensor] = {}
        if asnorm:
            if (not cohort_list) or (not cohort_path):
                raise RuntimeError("--asnorm requires --cohort_list and --cohort_path")

            # load or compute cohort embeddings (v1/v2)
            if cohort_emb_path and os.path.isfile(cohort_emb_path):
                npz = np.load(cohort_emb_path, allow_pickle=True)
                coh_v1 = torch.from_numpy(np.asarray(npz["v1"], dtype=np.float32)).cuda()  # [M,D]
                coh_v2 = torch.from_numpy(np.asarray(npz["v2"], dtype=np.float32)).cuda()  # [M,D]
            else:
                cohort_files = []
                for line in self._read_list(cohort_list):
                    ss = line.split()
                    if not ss:
                        continue
                    cohort_files.append(ss[-1])

                v1_list = []
                v2_list = []
                for cf in tqdm.tqdm(cohort_files, desc="CohortEmb", total=len(cohort_files)):
                    wav_path = cf if os.path.isabs(cf) else os.path.join(cohort_path, cf)
                    e1, e2 = self._embed_file(wav_path)
                    v1 = e1.squeeze(0)          # [D]
                    v2 = e2.mean(dim=0)         # [D] (exactly matches mean(score_2))
                    v1_list.append(v1.detach().cpu().numpy().astype(np.float32, copy=False))
                    v2_list.append(v2.detach().cpu().numpy().astype(np.float32, copy=False))

                coh_v1 = torch.from_numpy(np.stack(v1_list, axis=0)).cuda()
                coh_v2 = torch.from_numpy(np.stack(v2_list, axis=0)).cuda()

                if cohort_emb_path:
                    os.makedirs(os.path.dirname(cohort_emb_path), exist_ok=True)
                    np.savez(cohort_emb_path, v1=coh_v1.detach().cpu().numpy(), v2=coh_v2.detach().cpu().numpy())

            topk = int(asnorm_topk)
            eps = 1e-5

            for f in tqdm.tqdm(setfiles, desc="ASNormStats", total=len(setfiles)):
                e1, e2 = embeddings[f]
                v1 = e1.squeeze(0)        # [D]
                v2 = e2.mean(dim=0)       # [D]

                # cohort scores: s = 0.5*(coh_v1·v1 + coh_v2·v2)
                s_coh = 0.5 * (torch.matmul(coh_v1, v1) + torch.matmul(coh_v2, v2))  # [M]
                k = min(topk, int(s_coh.numel()))
                vals = torch.topk(s_coh, k=k, largest=True).values
                m = vals.mean()
                s = vals.std(unbiased=False).clamp_min(eps)
                mu[f] = m
                sd[f] = s

        # 3) score each trial
        scores: List[float] = []
        labels: List[int] = []

        for line in lines:
            ss = line.split()
            if len(ss) < 3:
                continue
            lab = int(ss[0])
            enr = ss[1]
            tst = ss[2]

            e11, e12 = embeddings[enr]
            e21, e22 = embeddings[tst]

            v1_enr = e11.squeeze(0)
            v2_enr = e12.mean(dim=0)
            v1_tst = e21.squeeze(0)
            v2_tst = e22.mean(dim=0)

            raw = 0.5 * (torch.dot(v1_enr, v1_tst) + torch.dot(v2_enr, v2_tst))

            if asnorm:
                raw = 0.5 * ((raw - mu[enr]) / sd[enr] + (raw - mu[tst]) / sd[tst])

            scores.append(float(raw.detach().cpu().item()))
            labels.append(lab)

        # 4) compute EER/minDCF
        scores_np = np.asarray(scores, dtype=np.float64)
        labels_np = np.asarray(labels, dtype=np.int64)

        try:
            from fair_utils import compute_eer, compute_minDCF
            eer, _ = compute_eer(scores_np, labels_np)
            mindcf = compute_minDCF(scores_np, labels_np, p_target=0.01)
            eer_pct = float(eer) * 100.0
        except Exception:
            eer_pct = float("nan")
            mindcf = float("nan")

        if return_scores:
            return eer_pct, float(mindcf), scores, labels
        return eer_pct, float(mindcf)

    def save_parameters(self, path: str):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path: str):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location="cpu")
        for name, param in loaded_state.items():
            orig = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print(f"{orig} is not in the model.")
                    continue
            if self_state[name].size() != param.size():
                print(f"Wrong parameter length: {orig}, model: {self_state[name].size()}, loaded: {param.size()}")
                continue
            self_state[name].copy_(param)
