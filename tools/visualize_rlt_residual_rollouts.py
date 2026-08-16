#!/usr/bin/env python3
"""Render RLT residual rollouts and action-delta diagnostics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw

from rlinf.algorithms.rlt.route import RLTRouteContext, build_rlt_route
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.models import get_model


RUN_PRESETS = {
    "gs4500": {
        "stage1_actor": "/data/RLinf/logs/stage1_pi05_4gpu_peg_resume_0123_20260801_172736/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_4500/actor",
        "stage2_ckpt": "/data/RLinf/logs/stage2_gs4500_res01_warm10k64_resume250_20260803_063929/maniskill_rlt_stage2_residual_entropy_ac_mlp/checkpoints/global_step_1200/actor/model_state_dict/full_weights.pt",
    },
    "gs10000": {
        "stage1_actor": "/data/RLinf/logs/stage1_pi05_4gpu_peg_resume_0123_20260801_172736/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_10000/actor",
        "stage2_ckpt": "/data/RLinf/logs/stage2_from_stage1_10000_always_res01_warm10k_64env_all7_20260803_133324/maniskill_rlt_stage2_residual_entropy_ac_mlp/checkpoints/global_step_800/actor/model_state_dict/full_weights.pt",
    },
}


@dataclass
class StepTrace:
    step: int
    actor_switch: bool
    reward: float
    success: bool
    done: bool
    actual_delta: np.ndarray
    student_delta: np.ndarray
    base_action: np.ndarray
    actual_action: np.ndarray
    student_action: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=sorted(RUN_PRESETS), default="gs4500")
    parser.add_argument("--stage1-actor", default=None)
    parser.add_argument("--stage2-ckpt", default=None)
    parser.add_argument("--out-dir", default="/data/RLinf/logs/visualizations/rlt_residual")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--interp-factor",
        type=int,
        default=3,
        help="Linear frame interpolation factor. 3 turns 10Hz env frames into 30fps video.",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--shader-pack", default="minimal")
    parser.add_argument("--config-name", default="maniskill_rlt_stage2_residual_entropy_ac_mlp")
    parser.add_argument("--config-dir", default="/data/RLinf/examples/embodiment/config")
    parser.add_argument(
        "--norm-stats-path",
        default="/data/datasets/lerobot/maniskill_peginsertionside_joint/norm_stats.json",
    )
    return parser.parse_args()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _first_scalar_bool(info: dict[str, Any], key: str) -> bool:
    if key not in info:
        return False
    value = _to_numpy(info[key]).reshape(-1)
    return bool(value[0]) if value.size else False


def _first_scalar_float(value: Any) -> float:
    arr = _to_numpy(value).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def _image_from_obs(obs: dict[str, Any]) -> np.ndarray:
    main = _to_numpy(obs["main_images"])[0, ..., :3].astype(np.uint8)
    wrist = None
    if "wrist_images" in obs and obs["wrist_images"] is not None:
        wrist = _to_numpy(obs["wrist_images"])[0, ..., :3].astype(np.uint8)
    if wrist is None:
        return main
    if wrist.shape[:2] != main.shape[:2]:
        wrist = np.asarray(Image.fromarray(wrist).resize((main.shape[1], main.shape[0])))
    gap = np.full((main.shape[0], 6, 3), 20, dtype=np.uint8)
    return np.concatenate([main, gap, wrist], axis=1)


def _blend_frames(frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
    if factor <= 1 or len(frames) <= 1:
        return frames
    output: list[np.ndarray] = []
    for left, right in zip(frames[:-1], frames[1:]):
        output.append(left)
        left_f = left.astype(np.float32)
        right_f = right.astype(np.float32)
        for i in range(1, factor):
            alpha = i / float(factor)
            output.append(((1.0 - alpha) * left_f + alpha * right_f).astype(np.uint8))
    output.append(frames[-1])
    return output


def _draw_text_bar(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame)
    bar_h = 34
    canvas = Image.new("RGB", (image.width, image.height + bar_h), (18, 18, 18))
    canvas.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 9), text, fill=(235, 235, 235))
    return np.asarray(canvas)


def _draw_curve_panel(
    *,
    width: int,
    height: int,
    traces: list[StepTrace],
    upto: int,
    y_max: float,
) -> np.ndarray:
    panel = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(panel)
    margin_l, margin_r, margin_t, margin_b = 44, 14, 32, 36
    x0, y0 = margin_l, margin_t
    x1, y1 = width - margin_r, height - margin_b
    draw.rectangle((x0, y0, x1, y1), outline=(50, 50, 50), width=1)
    draw.text((10, 8), "residual vs base", fill=(20, 20, 20))
    draw.text((x0, y1 + 10), "step", fill=(20, 20, 20))
    draw.text((8, y0 - 4), "L2", fill=(20, 20, 20))
    draw.text((x0 + 8, y0 + 8), "actual", fill=(34, 104, 185))
    draw.text((x0 + 72, y0 + 8), "student", fill=(205, 89, 34))
    if not traces:
        return np.asarray(panel)

    n = max(len(traces) - 1, 1)
    y_max = max(y_max, 1e-6)

    def xy(idx: int, value: float) -> tuple[int, int]:
        x = x0 + int((x1 - x0) * idx / n)
        y = y1 - int((y1 - y0) * min(max(value, 0.0), y_max) / y_max)
        return x, y

    actual = [float(np.linalg.norm(t.actual_delta)) for t in traces]
    student = [float(np.linalg.norm(t.student_delta)) for t in traces]
    upto = max(0, min(upto, len(traces) - 1))
    for idx, trace in enumerate(traces[: upto + 1]):
        if trace.actor_switch:
            x = xy(idx, 0.0)[0]
            draw.line((x, y0, x, y1), fill=(225, 240, 225), width=1)
    if upto > 0:
        draw.line([xy(i, actual[i]) for i in range(upto + 1)], fill=(34, 104, 185), width=3)
        draw.line([xy(i, student[i]) for i in range(upto + 1)], fill=(205, 89, 34), width=2)
    draw.text((x1 - 70, y0 + 8), f"max {y_max:.3f}", fill=(20, 20, 20))
    draw.text((x1 - 78, y1 + 10), f"{len(traces)}", fill=(20, 20, 20))
    return np.asarray(panel)


def _make_diagnostic_video(
    frames: list[np.ndarray],
    traces: list[StepTrace],
    output_path: Path,
    fps: int,
    interp_factor: int,
) -> None:
    if not frames:
        return
    y_max = max(
        [
            *(float(np.linalg.norm(t.actual_delta)) for t in traces),
            *(float(np.linalg.norm(t.student_delta)) for t in traces),
            1e-4,
        ]
    )
    annotated = []
    for idx, frame in enumerate(frames):
        trace_idx = min(idx, max(len(traces) - 1, 0))
        if traces:
            t = traces[trace_idx]
            text = (
                f"step={t.step:03d} switch={int(t.actor_switch)} "
                f"|actual-base|={np.linalg.norm(t.actual_delta):.4f} "
                f"reward={t.reward:.3f} success={int(t.success)}"
            )
        else:
            text = f"frame={idx:03d}"
        left = _draw_text_bar(frame, text)
        panel = _draw_curve_panel(
            width=360,
            height=left.shape[0],
            traces=traces,
            upto=trace_idx,
            y_max=y_max,
        )
        annotated.append(np.concatenate([left, panel], axis=1))

    annotated = _blend_frames(annotated, interp_factor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, macro_block_size=1)
    try:
        for frame in annotated:
            writer.append_data(frame)
    finally:
        writer.close()


def _save_static_plot(traces: list[StepTrace], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [t.step for t in traces]
    actual = np.stack([t.actual_delta for t in traces], axis=0)
    student = np.stack([t.student_delta for t in traces], axis=0)
    switch = np.asarray([t.actor_switch for t in traces], dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(steps, np.linalg.norm(actual, axis=1), label="||actual - base||")
    axes[0].plot(steps, np.linalg.norm(student, axis=1), label="||student - base||", alpha=0.7)
    axes[0].fill_between(
        steps,
        0,
        max(1e-6, float(np.linalg.norm(student, axis=1).max())),
        where=switch,
        color="green",
        alpha=0.08,
        label="actor_switch",
    )
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("L2 delta")

    for dim in range(actual.shape[1]):
        axes[1].plot(steps, actual[:, dim], label=f"a{dim}", linewidth=1)
    axes[1].set_ylabel("actual-base")
    axes[1].legend(ncol=4, fontsize=8)

    for dim in range(student.shape[1]):
        axes[2].plot(steps, student[:, dim], label=f"a{dim}", linewidth=1)
    axes[2].set_ylabel("student-base")
    axes[2].set_xlabel("env control step")
    axes[2].legend(ncol=4, fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_csv(traces: list[StepTrace], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "actor_switch",
        "reward",
        "success",
        "done",
        "actual_delta_l2",
        "student_delta_l2",
    ]
    for prefix in ("actual_delta", "student_delta", "base_action", "actual_action", "student_action"):
        fieldnames.extend(f"{prefix}_{i}" for i in range(8))
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in traces:
            row = {
                "step": t.step,
                "actor_switch": int(t.actor_switch),
                "reward": t.reward,
                "success": int(t.success),
                "done": int(t.done),
                "actual_delta_l2": float(np.linalg.norm(t.actual_delta)),
                "student_delta_l2": float(np.linalg.norm(t.student_delta)),
            }
            for prefix in ("actual_delta", "student_delta", "base_action", "actual_action", "student_action"):
                values = getattr(t, prefix)
                for i in range(8):
                    row[f"{prefix}_{i}"] = float(values[i])
            writer.writerow(row)


def _make_cfg(args: argparse.Namespace):
    preset = RUN_PRESETS[args.run]
    stage1_actor = args.stage1_actor or preset["stage1_actor"]
    stage2_ckpt = args.stage2_ckpt or preset["stage2_ckpt"]
    with initialize_config_dir(version_base="1.1", config_dir=args.config_dir):
        cfg = compose(
            config_name=args.config_name,
            overrides=[
                "runner.logger.logger_backends=[]",
                "runner.val_check_interval=0",
                "runner.save_interval=0",
                "cluster.component_placement.actor=0-0",
                "cluster.component_placement.env=0-0",
                "cluster.component_placement.rollout=0-0",
                "env.train.total_num_envs=1",
                "env.eval.total_num_envs=1",
                f"env.train.max_episode_steps={args.max_steps}",
                f"env.eval.max_episode_steps={args.max_steps}",
                f"env.train.max_steps_per_rollout_epoch={args.max_steps}",
                f"env.eval.max_steps_per_rollout_epoch={args.max_steps}",
                f"env.train.init_params.max_episode_steps={args.max_steps}",
                f"env.eval.init_params.max_episode_steps={args.max_steps}",
                f"env.train.init_params.sensor_configs.width={args.width}",
                f"env.train.init_params.sensor_configs.height={args.height}",
                f"env.eval.init_params.sensor_configs.width={args.width}",
                f"env.eval.init_params.sensor_configs.height={args.height}",
                f"env.train.init_params.sensor_configs.shader_pack={args.shader_pack}",
                f"env.eval.init_params.sensor_configs.shader_pack={args.shader_pack}",
                "env.train.rlt_policy_switch.trigger_mode=always_on",
                "env.eval.rlt_policy_switch.trigger_mode=always_on",
                "env.train.rlt_policy_switch.expert_takeover.enable=False",
                "env.eval.rlt_policy_switch.expert_takeover.enable=False",
                "rollout.expert_model=null",
                f"rollout.rlt_feature_model.model_path={stage1_actor}",
                f"rollout.rlt_feature_model.openpi_data.norm_stats_path={args.norm_stats_path}",
                "actor.model.residual_scale=0.1",
                "rollout.model.residual_scale=0.1",
            ],
        )
    with open_dict(cfg):
        cfg.env.eval.video_cfg.save_video = False
        cfg.env.eval.use_fixed_reset_state_ids = False
        cfg.runner.ckpt_path = stage2_ckpt
    return cfg, Path(stage2_ckpt), Path(stage1_actor)


def _load_models(cfg, ckpt_path: Path, device: torch.device):
    policy_cfg = copy.deepcopy(cfg.actor.model)
    with open_dict(policy_cfg):
        policy_cfg.precision = cfg.actor.model.precision
        policy_cfg.model_path = ""
    policy = get_model(policy_cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("[load] first missing:", missing[:8])
        if unexpected:
            print("[load] first unexpected:", unexpected[:8])
    policy.to(device)
    policy.eval()

    feature_model = get_model(copy.deepcopy(cfg.rollout.rlt_feature_model))
    feature_model.to(device)
    feature_model.eval()
    feature_model.requires_grad_(False)
    return policy, feature_model


def _new_env(cfg, seed: int):
    env_cfg = copy.deepcopy(cfg.env.eval)
    with open_dict(env_cfg):
        env_cfg.seed = seed
        env_cfg.total_num_envs = 1
        env_cfg.init_params.num_envs = 1
        env_cfg.video_cfg.save_video = False
    env_cls = get_env_cls(env_cfg.env_type, env_cfg)
    return env_cls(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def _run_one_rollout(
    *,
    cfg,
    policy,
    feature_model,
    env,
    route,
    device: torch.device,
    max_steps: int,
) -> tuple[list[np.ndarray], list[StepTrace], dict[str, Any]]:
    obs, infos = env.reset()
    frames = [_image_from_obs(obs)]
    traces: list[StepTrace] = []
    done = False
    step = 0
    ready_version = int(cfg.algorithm.rlt_schedule.get("warmup_post_collect_updates", 0))
    residual_scale = float(cfg.actor.model.get("residual_scale", 1.0))

    while step < max_steps and not done:
        with torch.no_grad():
            rlt_obs = feature_model.extract_rlt_obs(obs)
            student_actions, result = policy.predict_action_batch(
                env_obs=rlt_obs,
                mode="eval",
                return_obs=True,
            )
            route_output = route.route(
                RLTRouteContext(
                    env_obs=obs,
                    rlt_obs=rlt_obs,
                    student_actions=student_actions,
                    result=result,
                    mode="eval",
                    rlt_switch_flags=infos.get("rlt_switch_flags"),
                    intervene_requested=infos.get("intervene_flag"),
                    expert_model=None,
                    version=ready_version,
                )
            )
            routed_actions = route_output.actions

        base = rlt_obs["ref_chunk"][:, : routed_actions.shape[1], : routed_actions.shape[2]]
        actual_delta = (routed_actions - base).detach().cpu().float().numpy()[0]
        student_delta = (student_actions - base).detach().cpu().float().numpy()[0]
        base_np = base.detach().cpu().float().numpy()[0]
        actual_np = routed_actions.detach().cpu().float().numpy()[0]
        student_np = student_actions.detach().cpu().float().numpy()[0]

        exec_actions = prepare_actions(
            raw_chunk_actions=routed_actions,
            env_type=cfg.env.eval.env_type,
            model_type=cfg.actor.model.model_type,
            num_action_chunks=routed_actions.shape[1],
            action_dim=cfg.actor.model.action_dim,
            policy=cfg.actor.model.get("policy_setup", None),
            env_cfg=cfg.env.eval,
        )
        obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(exec_actions)

        actor_switch_tensor = route_output.result["forward_inputs"].get("actor_switch")
        actor_switch = _first_scalar_bool({"actor_switch": actor_switch_tensor}, "actor_switch")
        chunk_done = torch.logical_or(terminations, truncations)
        chunk_len = min(routed_actions.shape[1], max_steps - step)
        for i in range(chunk_len):
            info_i = infos_list[i]
            reward_i = _first_scalar_float(rewards[:, i])
            done_i = _first_scalar_bool({"done": chunk_done[:, i]}, "done")
            success_i = _first_scalar_bool(info_i, "success")
            traces.append(
                StepTrace(
                    step=step,
                    actor_switch=actor_switch,
                    reward=reward_i,
                    success=success_i,
                    done=done_i,
                    actual_delta=actual_delta[i].copy(),
                    student_delta=student_delta[i].copy() / max(residual_scale, 1e-12),
                    base_action=base_np[i].copy(),
                    actual_action=actual_np[i].copy(),
                    student_action=student_np[i].copy(),
                )
            )
            frames.append(_image_from_obs(obs_list[i]))
            step += 1
            if done_i:
                done = True
                break

        obs = obs_list[-1]
        infos = infos_list[-1]

    summary = {
        "steps": len(traces),
        "success": bool(any(t.success for t in traces)),
        "done": bool(traces[-1].done if traces else False),
        "actor_switch_rate": float(np.mean([t.actor_switch for t in traces])) if traces else 0.0,
        "actual_delta_l2_mean": float(np.mean([np.linalg.norm(t.actual_delta) for t in traces])) if traces else 0.0,
        "actual_delta_l2_max": float(np.max([np.linalg.norm(t.actual_delta) for t in traces])) if traces else 0.0,
        "student_delta_l2_mean": float(np.mean([np.linalg.norm(t.student_delta) for t in traces])) if traces else 0.0,
        "student_delta_l2_max": float(np.max([np.linalg.norm(t.student_delta) for t in traces])) if traces else 0.0,
    }
    return frames, traces, summary


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    cfg, ckpt_path, stage1_actor = _make_cfg(args)
    out_root = Path(args.out_dir) / args.run / f"{ckpt_path.parents[2].name}_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[config] run={args.run}")
    print(f"[config] stage1_actor={stage1_actor}")
    print(f"[config] stage2_ckpt={ckpt_path}")
    print(f"[config] out_root={out_root}")

    policy, feature_model = _load_models(cfg, ckpt_path, device)
    route = build_rlt_route(cfg)
    summaries = []
    for rollout_idx in range(args.num_rollouts):
        env = _new_env(cfg, seed=args.seed + rollout_idx)
        try:
            frames, traces, summary = _run_one_rollout(
                cfg=cfg,
                policy=policy,
                feature_model=feature_model,
                env=env,
                route=route,
                device=device,
                max_steps=args.max_steps,
            )
        finally:
            if hasattr(env, "close"):
                env.close()

        stem = f"rollout_{rollout_idx:02d}"
        video_path = out_root / f"{stem}_residual_diagnostics.mp4"
        plot_path = out_root / f"{stem}_residual_delta.png"
        csv_path = out_root / f"{stem}_residual_delta.csv"
        _make_diagnostic_video(
            frames=frames,
            traces=traces,
            output_path=video_path,
            fps=args.fps,
            interp_factor=args.interp_factor,
        )
        _save_static_plot(traces, plot_path)
        _save_csv(traces, csv_path)
        summary.update(
            {
                "rollout": rollout_idx,
                "video": str(video_path),
                "plot": str(plot_path),
                "csv": str(csv_path),
            }
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    metadata = {
        "run": args.run,
        "stage1_actor": str(stage1_actor),
        "stage2_ckpt": str(ckpt_path),
        "fps": args.fps,
        "interp_factor": args.interp_factor,
        "max_steps": args.max_steps,
        "summaries": summaries,
    }
    metadata_path = out_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[done] metadata={metadata_path}")


if __name__ == "__main__":
    main()
