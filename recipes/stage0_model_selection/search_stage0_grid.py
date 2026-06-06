#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import subprocess
from pathlib import Path

# =======================
# 你需要按实际情况改这里
# =======================
# 训练脚本：你现在用的是 trainECAPAModel_fair_ddp.py
SCRIPT = "trainECAPAModel_fair_ddp.py"

# 数据
TRAIN_LIST = "/medias/speech/projects/quy/dataset/voxceleb2_train_sub_list.txt"
TRAIN_PATH = "/medias/speech/data/VoxCeleb2"

# 评测协议（建议先 defer_eval，避免你现在 eval 的 padding/collate 问题）
EVAL_LISTS = [
    "/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-O-abs.txt",
    "/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-E-abs.txt",
    "/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/sub-vox1-H-abs.txt",
]
EVAL_PATH = "/medias/speech/data/VoxCeleb1"

GENDER_MAP = "/medias/speech/projects/quy/ECAPA-TDNN-all/ECAPA-TDNN-Paper/gender_map.json"
EVAL_GENDER_MAP = GENDER_MAP

# 训练公共超参
N_CLASS = 5994
C = 1024
M = 0.2
S = 30
MAX_EPOCH = 100
BATCH_SIZE = 1024
NUM_FRAMES = 200
LR = 1e-3
LR_DECAY = 1.0
AMP = True

SAVE_EVERY = 10
TEST_STEP = 9999999      # 训练时不做 inline eval（强烈建议先这样）
DEFER_EVAL = True

# DDP 资源
CUDA_VISIBLE_DEVICES = "0,1,2,3"
NPROC_PER_NODE = 4
MASTER_PORT_BASE = 29501   # 每个实验会在这个基础上 + idx，避免端口占用残留

# 输出根目录
OUT_ROOT = "exps/grid20_runs"


def build_cmd(cfg: dict, exp_dir: Path, idx: int):
    save_path = str(exp_dir)
    model_save_path = str(exp_dir / "models")

    # torchrun 部分：注意 `--` 后面紧跟脚本名，后面才是脚本参数
    cmd = [
        "torchrun",
        f"--nproc-per-node={NPROC_PER_NODE}",
        f"--master-port={MASTER_PORT_BASE + idx}",
        "--",
        SCRIPT,

        "--ddp", "--dist_backend", "nccl",

        "--train_list", TRAIN_LIST,
        "--train_path", TRAIN_PATH,

        "--eval_list", ",".join(EVAL_LISTS),
        "--eval_path", EVAL_PATH,
        "--gender_map", GENDER_MAP,
        "--eval_gender_map", EVAL_GENDER_MAP,

        "--save_path", save_path,
        "--model_save_path", model_save_path,

        "--n_class", str(N_CLASS),
        "--C", str(C),
        "--m", str(M),
        "--s", str(S),

        "--max_epoch", str(MAX_EPOCH),
        "--batch_size", str(BATCH_SIZE),
        "--num_frames", str(NUM_FRAMES),

        "--lr", str(LR),
        "--lr_decay", str(LR_DECAY),

        "--save_every", str(SAVE_EVERY),
        "--test_step", str(TEST_STEP),
    ]

    if AMP:
        cmd += ["--amp"]
    if DEFER_EVAL:
        cmd += ["--defer_eval"]

    # baseline 开关
    if cfg.get("baseline", False):
        cmd += ["--baseline"]

    # Fair/gate/adv 超参（无论 baseline 与否都传也没关系；baseline 模式下模型内部会忽略/不构建分支）
    cmd += ["--css_target_ratio", str(cfg.get("css_target_ratio", 0.8))]
    cmd += ["--lambda_css_cap", str(cfg.get("lambda_css_cap", 0.0))]
    cmd += ["--lambda_css_sat", str(cfg.get("lambda_css_sat", 0.0))]

    cmd += ["--lambda_gender_s", str(cfg.get("lambda_gender_s", 0.0))]
    cmd += ["--lambda_gender_adv", str(cfg.get("lambda_gender_adv", 0.0))]

    cmd += ["--lambda_decor", str(cfg.get("lambda_decor", 0.0))]
    cmd += ["--lambda_rex", str(cfg.get("lambda_rex", 0.0))]

    cmd += ["--lambda_fair_fmr", str(cfg.get("lambda_fair_fmr", 0.0))]

    cmd += ["--grl_lambda", str(cfg.get("grl_lambda", 0.0))]
    cmd += ["--grl_warmup_epochs", str(cfg.get("grl_warmup_epochs", 0))]
    cmd += ["--adv_warmup_epochs", str(cfg.get("adv_warmup_epochs", 0))]
    cmd += ["--adv_dropout", str(cfg.get("adv_dropout", 0.0))]
    cmd += ["--adv_noise_std", str(cfg.get("adv_noise_std", 0.0))]

    cmd += ["--stopgrad_gender_to_emb", str(cfg.get("stopgrad_gender_to_emb", 1))]

    return cmd


def main():
    here = Path(__file__).resolve().parent
    cfg_path = here / "grid_micro_1to10_percent_20.json"
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "r") as f:
        cfgs = json.load(f)

    print(f"[INFO] Loaded {len(cfgs)} configs from {cfg_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    env.setdefault("OMP_NUM_THREADS", "1")

    for idx, cfg in enumerate(cfgs):
        name = cfg.get("name", f"cfg_{idx:03d}")
        exp_dir = out_root / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "models").mkdir(parents=True, exist_ok=True)

        log_path = exp_dir / "train.log"
        cmd = build_cmd(cfg, exp_dir, idx)

        print("=" * 80)
        print(f"[RUN {idx+1:02d}/{len(cfgs)}] {name}")
        print(f"[SAVE] {exp_dir}")
        print(f"[LOG ] {log_path}")
        print("[CMD ] " + " ".join(cmd))

        t0 = time.time()
        with open(log_path, "w") as logf:
            logf.write("[CMD] " + " ".join(cmd) + "\n\n")
            logf.flush()
            subprocess.run(cmd, env=env, stdout=logf, stderr=logf, check=True)
        dt = time.time() - t0
        print(f"[DONE] {name} finished in {dt/3600.0:.2f} hours")

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
