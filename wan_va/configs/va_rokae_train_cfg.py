# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_rokae_cfg import va_rokae_cfg


va_rokae_train_cfg = EasyDict(__name__="Config: VA rokae train")
va_rokae_train_cfg.update(va_rokae_cfg)

va_rokae_train_cfg.dataset_path = os.path.abspath(
    os.getenv("ROKAE_DATASET_PATH", "data/rokae_demo/260514/lerobot/pick_up_the_workpiece")
)
va_rokae_train_cfg.empty_emb_path = os.path.join(va_rokae_train_cfg.dataset_path, "empty_emb.pt")
va_rokae_train_cfg.enable_wandb = False
va_rokae_train_cfg.load_worker = 8
va_rokae_train_cfg.save_interval = 500
va_rokae_train_cfg.gc_interval = 50
va_rokae_train_cfg.cfg_prob = 0.1
va_rokae_train_cfg.resume_from = None
va_rokae_train_cfg.enable_cpu_offload = os.getenv("ROKAE_ENABLE_CPU_OFFLOAD", "1") == "1"

# Training parameters
va_rokae_train_cfg.learning_rate = 1e-5
va_rokae_train_cfg.beta1 = 0.9
va_rokae_train_cfg.beta2 = 0.95
va_rokae_train_cfg.weight_decay = 0.1
va_rokae_train_cfg.warmup_steps = 10
va_rokae_train_cfg.batch_size = 1
va_rokae_train_cfg.gradient_accumulation_steps = 8
va_rokae_train_cfg.num_steps = 5000
va_rokae_train_cfg.save_root = os.path.abspath(os.getenv("ROKAE_TRAIN_ROOT", "runs/rokae_pick_up_the_workpiece"))
