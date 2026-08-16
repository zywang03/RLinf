#!/usr/bin/env python
"""Collect warmup RLT embeddings, train failure signal, and visualize them."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw

from rlinf.algorithms.rlt.failure_signal import (
    RLTFailureSignal,
    RLTFailureSignalEpisode,
    RLTFailureSignalTrainer,
)
from rlinf.envs.maniskill.maniskill_rlt_env import ManiskillRLTEnv
from rlinf.models import get_model


@dataclass
class ProbeEpisode:
    episode_id: int
    env_id: int
    z_list: list[torch.Tensor] = field(default_factory=list)
    image_list: list[np.ndarray] = field(default_factory=list)
    step_id_list: list[int] = field(default_factory=list)
    reward_sum: float = 0.0
    success: bool = False
    done: bool = False
    truncated_by_probe: bool = False

    def as_failure_signal_episode(self) -> RLTFailureSignalEpisode:
        z_rl = torch.stack(self.z_list, dim=0).float()
        return RLTFailureSignalEpisode(z_rl=z_rl, failed=not self.success)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="maniskill_rlt_stage2_ac_mlp")
    parser.add_argument(
        "--base-actor-path",
        default="/data/RLinf/logs/pi_base_model/global_step_10000/actor",
    )
    parser.add_argument(
        "--norm-stats-path",
        default="/data/datasets/lerobot/maniskill_peginsertionside_joint/norm_stats.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to logs/visualizations/rlt_failure_signal_probe_<timestamp>.",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--target-episodes", type=int, default=24)
    parser.add_argument("--min-success", type=int, default=2)
    parser.add_argument("--min-failure", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=12)
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument(
        "--dense-stride-env-steps",
        type=int,
        default=-1,
        help=(
            "-1 disables dense probing; 0 probes every env step inside a chunk; "
            "N probes every N env steps in addition to chunk starts."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--shader-pack", default="minimal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-saved-failure-images", type=int, default=160)
    parser.add_argument("--failure-score-threshold", type=float, default=0.7)
    parser.add_argument("--failure-distance-threshold", type=float, default=6.0)
    parser.add_argument("--max-centers", type=int, default=256)
    parser.add_argument("--pre-window", type=int, default=2)
    parser.add_argument("--post-window", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--min-train-steps", type=int, default=50)
    parser.add_argument("--max-train-steps", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    return parser.parse_args()


def build_cfg(args: argparse.Namespace) -> Any:
    config_dir = str(Path(__file__).resolve().parent / "config")
    overrides = [
        "runner.logger.logger_backends=[]",
        f"env.train.total_num_envs={args.num_envs}",
        f"env.train.seed={args.seed}",
        f"env.train.init_params.sensor_configs.width={args.width}",
        f"env.train.init_params.sensor_configs.height={args.height}",
        f"env.train.init_params.sensor_configs.shader_pack={args.shader_pack}",
        f"env.train.max_episode_steps={args.max_chunks * args.chunk_len}",
        f"env.train.init_params.max_episode_steps={args.max_chunks * args.chunk_len}",
        f"rollout.rlt_feature_model.model_path={args.base_actor_path}",
        f"rollout.rlt_feature_model.openpi_data.norm_stats_path={args.norm_stats_path}",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        out_dir = Path("/data/RLinf/logs/visualizations") / (
            f"rlt_failure_signal_probe_{stamp}"
        )
    (out_dir / "failed_obs_images").mkdir(parents=True, exist_ok=True)
    return out_dir


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def info_success(infos: dict[str, Any], rewards: torch.Tensor, env_idx: int) -> bool:
    for key in ("success", "success_current", "success_once"):
        value = infos.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > env_idx:
            return bool(value.reshape(value.shape[0], -1)[env_idx].any().item())
    return bool((rewards[env_idx] > 0).any().item())


def append_probe_sample(
    episode: ProbeEpisode,
    *,
    z_rl: torch.Tensor,
    image: torch.Tensor,
    step_id: int,
) -> None:
    episode.z_list.append(z_rl.detach().float().cpu())
    episode.image_list.append(tensor_image_to_uint8(image.detach().cpu()))
    episode.step_id_list.append(int(step_id))


def dense_probe_offsets(args: argparse.Namespace) -> list[int]:
    """Return 1-indexed env-step offsets inside a chunk to probe after stepping."""
    stride = int(args.dense_stride_env_steps)
    if stride < 0:
        return []
    if stride == 0:
        return list(range(1, args.chunk_len + 1))
    return list(range(stride, args.chunk_len + 1, stride))


def collect_episodes(
    *,
    env: ManiskillRLTEnv,
    feature_model: Any,
    args: argparse.Namespace,
) -> list[ProbeEpisode]:
    episodes: list[ProbeEpisode] = []
    next_episode_id = 0

    for batch_idx in range(args.max_batches):
        obs, _ = env.reset(seed=args.seed + batch_idx)
        active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        batch_eps = [
            ProbeEpisode(episode_id=next_episode_id + i, env_id=i)
            for i in range(args.num_envs)
        ]
        next_episode_id += args.num_envs

        offsets = dense_probe_offsets(args)
        for chunk_idx in range(args.max_chunks):
            with torch.inference_mode():
                rlt_obs = feature_model.extract_rlt_obs(obs)
            z_rl = rlt_obs["z_rl"].detach().float().cpu()
            ref_chunk = rlt_obs["ref_chunk"]
            if ref_chunk.dim() == 2:
                ref_chunk = ref_chunk.reshape(
                    ref_chunk.shape[0], args.chunk_len, args.action_dim
                )
            actions = ref_chunk[:, : args.chunk_len, : args.action_dim].contiguous()
            main_images = obs["main_images"].detach().cpu()
            chunk_start_step = chunk_idx * args.chunk_len

            for env_idx in range(args.num_envs):
                if not bool(active[env_idx].item()):
                    continue
                append_probe_sample(
                    batch_eps[env_idx],
                    z_rl=z_rl[env_idx],
                    image=main_images[env_idx],
                    step_id=chunk_start_step,
                )

            obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
                actions
            )
            infos = infos_list[-1]
            done = (terminations.any(dim=1) | truncations.any(dim=1)).to(active.device)
            obs = obs_list[-1]
            chunk_dones = (terminations | truncations).to(active.device)

            for offset in offsets:
                obs_idx = offset - 1
                if obs_idx < 0 or obs_idx >= len(obs_list):
                    continue
                done_before = (
                    chunk_dones[:, :obs_idx].any(dim=1)
                    if obs_idx > 0
                    else torch.zeros_like(active)
                )
                sample_mask = active & (~done_before)
                if not bool(sample_mask.any().item()):
                    continue
                with torch.inference_mode():
                    dense_rlt_obs = feature_model.extract_rlt_obs(obs_list[obs_idx])
                dense_z_rl = dense_rlt_obs["z_rl"].detach().float().cpu()
                dense_images = obs_list[obs_idx]["main_images"].detach().cpu()
                step_id = chunk_start_step + offset
                for env_idx in range(args.num_envs):
                    if not bool(sample_mask[env_idx].item()):
                        continue
                    append_probe_sample(
                        batch_eps[env_idx],
                        z_rl=dense_z_rl[env_idx],
                        image=dense_images[env_idx],
                        step_id=step_id,
                    )

            for env_idx in range(args.num_envs):
                if not bool(active[env_idx].item()):
                    continue
                ep = batch_eps[env_idx]
                ep.reward_sum += float(rewards[env_idx].detach().float().sum().item())
                ep.success = ep.success or info_success(infos, rewards, env_idx)
                if bool(done[env_idx].item()):
                    ep.done = True
                    episodes.append(ep)
                    active[env_idx] = False

            success_count = sum(int(ep.success) for ep in episodes)
            failure_count = sum(int(not ep.success) for ep in episodes)
            if (
                len(episodes) >= args.target_episodes
                and success_count >= args.min_success
                and failure_count >= args.min_failure
            ):
                return episodes
            if not bool(active.any().item()):
                break

        for env_idx in range(args.num_envs):
            if bool(active[env_idx].item()) and batch_eps[env_idx].z_list:
                batch_eps[env_idx].truncated_by_probe = True
                episodes.append(batch_eps[env_idx])

    return episodes


def train_failure_signal(
    episodes: list[ProbeEpisode], args: argparse.Namespace, device: torch.device
) -> tuple[RLTFailureSignal, dict[str, float]]:
    input_dim = int(episodes[0].z_list[0].numel())
    module = RLTFailureSignal(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        max_centers=args.max_centers,
        score_threshold=args.failure_score_threshold,
        distance_threshold=args.failure_distance_threshold,
    ).to(device)
    trainer = RLTFailureSignalTrainer(
        module,
        {
            "min_failure_episodes": 1,
            "min_success_episodes": 0,
            "min_steps": args.min_train_steps,
            "max_steps": args.max_train_steps,
            "patience": args.patience,
            "pre_window": args.pre_window,
            "post_window": args.post_window,
            "smoothness_weight": 1.0e-3,
        },
        device=device,
    )
    trainer.add_episodes([ep.as_failure_signal_episode() for ep in episodes])
    metrics = trainer.train_to_convergence()
    return module, metrics


def flatten_points(
    episodes: list[ProbeEpisode], module: RLTFailureSignal, device: torch.device
) -> dict[str, Any]:
    zs = []
    episode_ids = []
    env_ids = []
    time_ids = []
    failed = []
    success = []
    for ep in episodes:
        for step_idx, z in enumerate(ep.z_list):
            zs.append(z.reshape(-1))
            episode_ids.append(ep.episode_id)
            env_ids.append(ep.env_id)
            if ep.step_id_list:
                time_ids.append(ep.step_id_list[step_idx])
            else:
                time_ids.append(step_idx)
            failed.append(not ep.success)
            success.append(ep.success)
    z = torch.stack(zs, dim=0).float()
    with torch.inference_mode():
        pred = module.predict(z.to(device))
        x = module.transform(z.to(device)).detach().cpu()
    return {
        "z": z.numpy(),
        "x": x.numpy(),
        "episode_ids": np.asarray(episode_ids),
        "env_ids": np.asarray(env_ids),
        "time_ids": np.asarray(time_ids),
        "failed": np.asarray(failed, dtype=bool),
        "success": np.asarray(success, dtype=bool),
        "score": pred["score"].detach().cpu().reshape(-1).numpy(),
        "distance": pred["distance"].detach().cpu().reshape(-1).numpy(),
        "actor_switch": pred["actor_switch"].detach().cpu().reshape(-1).numpy(),
    }


def embed_points(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, str]:
    combined = np.concatenate([x, centers], axis=0) if len(centers) else x
    if combined.shape[0] < 3:
        return np.zeros((combined.shape[0], 2), dtype=np.float32), "constant"
    try:
        from sklearn.manifold import TSNE

        perplexity = max(2, min(30, (combined.shape[0] - 1) // 3))
        emb = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=0,
        ).fit_transform(combined)
        return emb, "tsne"
    except Exception:
        centered = combined - combined.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        emb = centered @ vt[:2].T
        return emb, "pca_fallback"


def save_plot(points: dict[str, Any], module: RLTFailureSignal, out_dir: Path) -> str:
    center_count = int(module.center_count.detach().cpu().item())
    centers = (
        module.failure_centers[:center_count].detach().float().cpu().numpy()
        if center_count > 0
        else np.zeros((0, points["x"].shape[1]), dtype=np.float32)
    )
    emb, method = embed_points(points["x"], centers)
    n = points["x"].shape[0]
    point_emb = emb[:n]
    center_emb = emb[n:]

    plt.figure(figsize=(9, 7))
    success_mask = points["success"]
    failed_mask = points["failed"]
    switched = points["actor_switch"].astype(bool)
    plt.scatter(
        point_emb[success_mask, 0],
        point_emb[success_mask, 1],
        c="#2f80ed",
        s=18,
        alpha=0.7,
        label="success z_rl",
    )
    plt.scatter(
        point_emb[failed_mask, 0],
        point_emb[failed_mask, 1],
        c="#eb5757",
        s=18,
        alpha=0.75,
        label="failed z_rl",
    )
    if switched.any():
        plt.scatter(
            point_emb[switched, 0],
            point_emb[switched, 1],
            facecolors="none",
            edgecolors="#111111",
            s=48,
            linewidths=0.8,
            label="gate on",
        )
    if len(center_emb):
        plt.scatter(
            center_emb[:, 0],
            center_emb[:, 1],
            c="#111111",
            marker="x",
            s=55,
            linewidths=1.2,
            label="failure centers",
        )
    plt.title(f"RLT failure signal embedding ({method})")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(loc="best")
    plt.tight_layout()
    path = out_dir / f"z_rl_{method}.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return method


def save_failed_images(
    episodes: list[ProbeEpisode],
    points: dict[str, Any],
    out_dir: Path,
    max_images: int,
) -> tuple[int, dict[int, int]]:
    image_dir = out_dir / "failed_obs_images"
    saved_tiles = []
    saved = 0
    onset_by_episode = failure_onsets(points)
    score_lookup = {
        (int(eid), int(tid)): float(score)
        for eid, tid, score in zip(
            points["episode_ids"], points["time_ids"], points["score"], strict=True
        )
    }
    for ep in episodes:
        if ep.success:
            continue
        for step_idx, image in enumerate(ep.image_list):
            if saved >= max_images:
                break
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            score = score_lookup.get((ep.episode_id, step_idx), float("nan"))
            is_onset = onset_by_episode.get(ep.episode_id) == step_idx
            if is_onset:
                for inset in range(5):
                    draw.rectangle(
                        (
                            inset,
                            inset,
                            pil.width - 1 - inset,
                            pil.height - 1 - inset,
                        ),
                        outline=(255, 230, 0),
                    )
            label = f"ep {ep.episode_id} t {step_idx} s {score:.2f}"
            if is_onset:
                label += " ONSET"
            draw.rectangle((0, 0, min(260, pil.width), 24), fill=(0, 0, 0))
            draw.text(
                (4, 4),
                label,
                fill=(255, 255, 255),
            )
            pil.save(image_dir / f"failed_ep{ep.episode_id:04d}_t{step_idx:03d}.png")
            saved_tiles.append(pil.resize((160, 160)))
            saved += 1
        if saved >= max_images:
            break

    if saved_tiles:
        cols = min(8, len(saved_tiles))
        rows = int(math.ceil(len(saved_tiles) / cols))
        sheet = Image.new("RGB", (cols * 160, rows * 160), color=(255, 255, 255))
        for idx, tile in enumerate(saved_tiles):
            sheet.paste(tile, ((idx % cols) * 160, (idx // cols) * 160))
        sheet.save(out_dir / "failed_obs_contact_sheet.png")
    return saved, onset_by_episode


def failure_onsets(points: dict[str, Any]) -> dict[int, int]:
    onset_by_episode: dict[int, int] = {}
    episode_ids = np.unique(points["episode_ids"][points["failed"]])
    for episode_id in episode_ids.tolist():
        mask = points["episode_ids"] == episode_id
        order = np.argsort(points["time_ids"][mask])
        times = points["time_ids"][mask][order]
        scores = points["score"][mask][order]
        if len(times) == 0:
            continue
        if len(times) == 1:
            onset_by_episode[int(episode_id)] = int(times[0])
            continue
        onset_pos = int(np.argmax(scores[1:] - scores[:-1]) + 1)
        onset_by_episode[int(episode_id)] = int(times[onset_pos])
    return onset_by_episode


def main() -> None:
    args = parse_args()
    out_dir = make_output_dir(args)
    cfg = build_cfg(args)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    feature_model = get_model(cfg.rollout.rlt_feature_model)
    feature_model.to(device)
    feature_model.eval()
    feature_model.requires_grad_(False)

    env = ManiskillRLTEnv(
        cfg.env.train,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )

    try:
        episodes = collect_episodes(env=env, feature_model=feature_model, args=args)
    finally:
        env.close()

    success_count = sum(int(ep.success) for ep in episodes)
    failure_count = sum(int(not ep.success) for ep in episodes)
    if failure_count == 0:
        raise RuntimeError("Collected no failed episodes; increase --target-episodes.")
    if not episodes or not episodes[0].z_list:
        raise RuntimeError("Collected no RLT embeddings.")

    module, metrics = train_failure_signal(episodes, args, device)
    points = flatten_points(episodes, module, device)
    center_count = int(module.center_count.detach().cpu().item())
    method = save_plot(points, module, out_dir)
    saved_images, onset_by_episode = save_failed_images(
        episodes, points, out_dir, args.max_saved_failure_images
    )
    with open(out_dir / "failure_onsets.json", "w") as f:
        json.dump(
            {str(k): int(v) for k, v in sorted(onset_by_episode.items())},
            f,
            indent=2,
        )

    np.savez_compressed(
        out_dir / "z_rl_points.npz",
        **points,
        failure_centers=module.failure_centers[:center_count]
        .detach()
        .float()
        .cpu()
        .numpy(),
        normalizer_mean=module.normalizer_mean.detach().float().cpu().numpy(),
        normalizer_std=module.normalizer_std.detach().float().cpu().numpy(),
    )
    torch.save(
        {
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "env_id": ep.env_id,
                    "success": ep.success,
                    "reward_sum": ep.reward_sum,
                    "truncated_by_probe": ep.truncated_by_probe,
                    "step_ids": list(ep.step_id_list),
                    "z_rl": torch.stack(ep.z_list, dim=0),
                }
                for ep in episodes
            ],
            "failure_signal_state_dict": module.state_dict(),
            "metrics": metrics,
        },
        out_dir / "failure_signal_probe.pt",
    )

    summary = {
        "output_dir": str(out_dir),
        "episodes": len(episodes),
        "success_episodes": success_count,
        "failed_episodes": failure_count,
        "z_points": int(points["z"].shape[0]),
        "center_count": center_count,
        "embedding_method": method,
        "saved_failure_images": saved_images,
        "dense_stride_env_steps": int(args.dense_stride_env_steps),
        "failure_onsets": {str(k): int(v) for k, v in sorted(onset_by_episode.items())},
        "failure_signal_metrics": metrics,
        "score_mean": float(np.mean(points["score"])),
        "gate_on_rate": float(np.mean(points["actor_switch"])),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
