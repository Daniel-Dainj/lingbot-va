#!/usr/bin/env python3
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Extract Wan2.2 VAE latents and text embeddings for the converted ROKAE dataset."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from transformers import T5TokenizerFast, UMT5EncoderModel

from wan_va.modules.utils import WanVAEStreamingWrapper, load_vae


OBS_CAM_TO_H5_KEY = {
    "observation.images.exterior_left": "observation/exterior_image_1_left",
    "observation.images.wrist": "observation/wrist_image",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing episode_*.h5 files")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Converted LeRobot dataset root")
    parser.add_argument("--model-root", type=Path, required=True, help="LingBot-VA base checkpoint root")
    parser.add_argument("--height", type=int, default=256, help="Per-camera resize height before VAE encoding")
    parser.add_argument("--width", type=int, default=320, help="Per-camera resize width before VAE encoding")
    parser.add_argument(
        "--obs-cam-key",
        action="append",
        default=["observation.images.exterior_left", "observation.images.wrist"],
        help="Camera feature name stored in the LeRobot dataset",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device used for encoding")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Text embedding sequence length")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap used for smoke tests")
    parser.add_argument("--skip-existing", action="store_true", help="Skip latent files that already exist")
    return parser.parse_args()


def load_episode_records(dataset_root: Path):
    records = []
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_source_manifest(dataset_root: Path):
    manifest_path = dataset_root / "meta" / "rokae_source_manifest.json"
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {int(item["episode_index"]): item for item in manifest}


def get_prompt_embedding(
    tokenizer: T5TokenizerFast,
    text_encoder: UMT5EncoderModel,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    text_inputs = tokenizer(
        [prompt_clean(prompt)],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_len = int(attention_mask.gt(0).sum(dim=1)[0].item())
    hidden_state = text_encoder(input_ids, attention_mask).last_hidden_state[0].to(dtype=dtype)
    out = hidden_state.new_zeros((max_sequence_length, hidden_state.shape[-1]))
    out[:seq_len] = hidden_state[:seq_len]
    return out.cpu()


def normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    return ((latents.float() - latents_mean) * latents_std).to(latents.dtype)


def resize_video(video_frames: np.ndarray, height: int, width: int) -> torch.Tensor:
    video = torch.from_numpy(video_frames).float().permute(3, 0, 1, 2)
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    return video.unsqueeze(0)


def save_empty_embedding(empty_emb: torch.Tensor, dataset_root: Path):
    empty_emb_path = dataset_root / "empty_emb.pt"
    if not empty_emb_path.exists():
        torch.save(empty_emb.to(torch.bfloat16), empty_emb_path)


def main():
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")

    device = torch.device(args.device)
    dtype = torch.bfloat16

    tokenizer = T5TokenizerFast.from_pretrained(model_root / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_root / "text_encoder", torch_dtype=dtype).to(device)
    text_encoder.eval()

    vae = load_vae(model_root / "vae", torch_dtype=dtype, torch_device=device)
    vae.eval()
    streaming_vae = WanVAEStreamingWrapper(vae)

    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype)

    empty_emb = get_prompt_embedding(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt="",
        device=device,
        dtype=dtype,
        max_sequence_length=args.max_seq_len,
    )
    save_empty_embedding(empty_emb, dataset_root)

    episode_records = load_episode_records(dataset_root)
    source_manifest = load_source_manifest(dataset_root)
    if args.max_episodes is not None:
        episode_records = episode_records[: args.max_episodes]

    for episode_record in episode_records:
        episode_index = episode_record["episode_index"]
        manifest_item = source_manifest.get(int(episode_index))
        if manifest_item is not None:
            episode_h5_path = raw_root / manifest_item["episode_file"]
        else:
            episode_h5_path = raw_root / f"episode_{episode_index:06d}.h5"
        if not episode_h5_path.exists():
            raise FileNotFoundError(f"Missing raw episode file: {episode_h5_path}")

        with h5py.File(episode_h5_path, "r") as h5_file:
            ori_fps = int(h5_file.attrs["fps"])
            for action_cfg in episode_record["action_config"]:
                start_frame = int(action_cfg["start_frame"])
                end_frame = int(action_cfg["end_frame"])
                action_text = action_cfg["action_text"]
                frame_ids = list(range(start_frame, end_frame))
                text_emb = get_prompt_embedding(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=action_text,
                    device=device,
                    dtype=dtype,
                    max_sequence_length=args.max_seq_len,
                ).to(torch.bfloat16)

                videos = []
                for obs_cam_key in args.obs_cam_key:
                    h5_key = OBS_CAM_TO_H5_KEY[obs_cam_key]
                    videos.append(resize_video(h5_file[h5_key][frame_ids], args.height, args.width))

                all_exist = True
                episode_chunk = episode_index // 1000
                output_paths = []
                for obs_cam_key in args.obs_cam_key:
                    latent_dir = dataset_root / "latents" / f"chunk-{episode_chunk:03d}" / obs_cam_key
                    latent_dir.mkdir(parents=True, exist_ok=True)
                    latent_path = latent_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                    output_paths.append((obs_cam_key, latent_path))
                    all_exist = all_exist and latent_path.exists()

                if args.skip_existing and all_exist:
                    continue

                video_batch = torch.cat(videos, dim=0).to(device=device, dtype=dtype)
                video_batch = video_batch / 255.0 * 2.0 - 1.0

                streaming_vae.clear_cache()
                with torch.inference_mode():
                    enc_out = streaming_vae.encode_chunk(video_batch)
                    mu, _ = torch.chunk(enc_out, 2, dim=1)
                    mu_norm = normalize_latents(mu, latents_mean, 1.0 / latents_std)

                for cam_idx, (obs_cam_key, latent_path) in enumerate(output_paths):
                    cam_latent = mu_norm[cam_idx : cam_idx + 1]
                    _, channels, latent_num_frames, latent_height, latent_width = cam_latent.shape
                    latent_payload = {
                        "latent": rearrange(cam_latent[0].detach().cpu(), "c f h w -> (f h w) c").to(torch.bfloat16),
                        "latent_num_frames": int(latent_num_frames),
                        "latent_height": int(latent_height),
                        "latent_width": int(latent_width),
                        "video_num_frames": int(len(frame_ids)),
                        "video_height": int(h5_file[OBS_CAM_TO_H5_KEY[obs_cam_key]].shape[1]),
                        "video_width": int(h5_file[OBS_CAM_TO_H5_KEY[obs_cam_key]].shape[2]),
                        "text_emb": text_emb,
                        "text": action_text,
                        "frame_ids": frame_ids,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "fps": ori_fps,
                        "ori_fps": ori_fps,
                        "channels": int(channels),
                    }
                    torch.save(latent_payload, latent_path)

    print(f"Saved empty embedding to {dataset_root / 'empty_emb.pt'}")
    print(f"Saved latent files under {dataset_root / 'latents'}")


if __name__ == "__main__":
    main()
