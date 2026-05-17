# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg


def _build_inverse_used_action_channel_ids(used_action_channel_ids, action_dim):
    inverse_used_action_channel_ids = [len(used_action_channel_ids)] * action_dim
    for i, j in enumerate(used_action_channel_ids):
        inverse_used_action_channel_ids[j] = i
    return inverse_used_action_channel_ids


va_rokae_cfg = EasyDict(__name__="Config: VA rokae")
va_rokae_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = "server"

va_rokae_cfg.wan22_pretrained_model_name_or_path = os.path.abspath(
    os.getenv("ROKAE_MODEL_PATH", "checkpoints/lingbot-va/lingbot-va-base")
)

va_rokae_cfg.attn_window = 30
va_rokae_cfg.frame_chunk_size = 4
va_rokae_cfg.env_type = "none"

va_rokae_cfg.height = 256
va_rokae_cfg.width = 320
va_rokae_cfg.action_dim = 30
va_rokae_cfg.action_per_frame = 4
va_rokae_cfg.obs_cam_keys = [
    "observation.images.exterior_left",
    "observation.images.wrist",
]
va_rokae_cfg.guidance_scale = 5
va_rokae_cfg.action_guidance_scale = 1

va_rokae_cfg.num_inference_steps = 5
va_rokae_cfg.video_exec_step = -1
va_rokae_cfg.action_num_inference_steps = 10

va_rokae_cfg.snr_shift = 5.0
va_rokae_cfg.action_snr_shift = 1.0

# Raw action order in the converted dataset:
# [left_eef_xyzquat(7), left_joint_position(7), left_gripper(1)]
va_rokae_cfg.used_action_channel_ids = list(range(0, 7)) + list(range(14, 21)) + [28]
va_rokae_cfg.inverse_used_action_channel_ids = _build_inverse_used_action_channel_ids(
    va_rokae_cfg.used_action_channel_ids, va_rokae_cfg.action_dim
)

va_rokae_cfg.action_norm_method = "quantiles"
va_rokae_cfg.norm_stat = {
    "q01": [0.0] * va_rokae_cfg.action_dim,
    "q99": [1.0] * va_rokae_cfg.action_dim,
}
