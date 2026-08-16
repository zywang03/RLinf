#!/usr/bin/env python3
"""Plot residual-vs-base action deltas from saved RLT replay transitions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch


RUN_PRESETS = {
    "gs4500": "/data/RLinf/logs/stage2_gs4500_res01_warm10k64_resume250_20260803_063929/maniskill_rlt_stage2_residual_entropy_ac_mlp/checkpoints/global_step_1200/actor/sac_components/replay_buffer/rank_0",
    "gs10000": "/data/RLinf/logs/stage2_from_stage1_10000_always_res01_warm10k_64env_all7_20260803_133324/maniskill_rlt_stage2_residual_entropy_ac_mlp/checkpoints/global_step_800/actor/sac_components/replay_buffer/rank_0",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=sorted(RUN_PRESETS), default="gs4500")
    parser.add_argument("--replay-dir", default=None)
    parser.add_argument("--out-dir", default="/data/RLinf/logs/visualizations/rlt_replay_deltas")
    parser.add_argument("--num-transitions", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    return parser.parse_args()


def _trajectory_id(path: Path) -> int:
    match = re.search(r"trajectory_(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def _latest_files(replay_dir: Path, count: int) -> list[Path]:
    files = sorted(replay_dir.glob("trajectory_*.pt"), key=_trajectory_id)
    return files[-count:]


def _tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _load_row(path: Path, action_dim: int, residual_scale: float) -> dict:
    data = torch.load(path, map_location="cpu")
    fwd = data["forward_inputs"]
    actual = _tensor(fwd["action"]).reshape(-1, action_dim)
    student = _tensor(fwd["model_action"]).reshape(-1, action_dim)
    base = _tensor(fwd["ref_chunk"]).reshape(-1, action_dim)[: actual.shape[0]]
    actor_switch = bool(_tensor(fwd["actor_switch"]).reshape(-1)[0].item())
    record_transition = bool(_tensor(fwd["record_transition"]).reshape(-1)[0].item())
    rewards = _tensor(data["rewards"]).reshape(-1)
    dones = _tensor(data["dones"]).reshape(-1).bool()
    actual_delta = actual - base
    student_delta = (student - base) / max(residual_scale, 1e-12)
    return {
        "path": str(path),
        "trajectory_id": _trajectory_id(path),
        "actor_switch": actor_switch,
        "record_transition": record_transition,
        "reward_sum": float(rewards.sum().item()),
        "done": bool(dones.any().item()),
        "actual_delta": actual_delta.numpy(),
        "student_delta": student_delta.numpy(),
        "actual": actual.numpy(),
        "student": student.numpy(),
        "base": base.numpy(),
    }


def _flatten_rows(rows: list[dict]) -> list[dict]:
    flat = []
    step = 0
    for row in rows:
        chunk_len = row["actual_delta"].shape[0]
        for i in range(chunk_len):
            actual_delta = row["actual_delta"][i]
            student_delta = row["student_delta"][i]
            item = {
                "step": step,
                "trajectory_id": row["trajectory_id"],
                "chunk_step": i,
                "actor_switch": int(row["actor_switch"]),
                "record_transition": int(row["record_transition"]),
                "reward_sum": row["reward_sum"],
                "done": int(row["done"]),
                "actual_delta_l2": float(np.linalg.norm(actual_delta)),
                "student_delta_l2": float(np.linalg.norm(student_delta)),
            }
            for prefix in ("actual_delta", "student_delta", "actual", "student", "base"):
                values = row[prefix][i]
                for dim, value in enumerate(values):
                    item[f"{prefix}_{dim}"] = float(value)
            flat.append(item)
            step += 1
    return flat


def _save_csv(flat: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(flat[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


def _save_plot(flat: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([r["step"] for r in flat])
    actual_l2 = np.asarray([r["actual_delta_l2"] for r in flat])
    student_l2 = np.asarray([r["student_delta_l2"] for r in flat])
    actor_switch = np.asarray([r["actor_switch"] for r in flat], dtype=bool)
    actual_dims = np.asarray(
        [[r[f"actual_delta_{i}"] for i in range(8)] for r in flat], dtype=np.float32
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    ymax = max(float(student_l2.max()), float(actual_l2.max()), 1e-6)
    axes[0].plot(steps, actual_l2, label="||actual - base||", linewidth=2)
    axes[0].plot(steps, student_l2, label="||(student - base) / scale||", alpha=0.75)
    axes[0].fill_between(steps, 0, ymax, where=actor_switch, color="green", alpha=0.08)
    axes[0].set_ylabel("delta L2")
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)

    for i in range(actual_dims.shape[1]):
        axes[1].plot(steps, actual_dims[:, i], label=f"a{i}", linewidth=1)
    axes[1].set_ylabel("actual - base")
    axes[1].legend(ncol=4, fontsize=8)

    chunk_starts = sorted({r["step"] for r in flat if r["chunk_step"] == 0})
    axes[2].plot(steps, [r["reward_sum"] for r in flat], label="chunk reward sum")
    axes[2].plot(steps, [r["done"] for r in flat], label="chunk done")
    for x in chunk_starts:
        axes[2].axvline(x, color="black", alpha=0.08, linewidth=1)
    axes[2].set_ylabel("chunk stats")
    axes[2].set_xlabel("control step across latest replay transitions")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    replay_dir = Path(args.replay_dir or RUN_PRESETS[args.run])
    rows = [
        _load_row(path, args.action_dim, args.residual_scale)
        for path in _latest_files(replay_dir, args.num_transitions)
    ]
    flat = _flatten_rows(rows)
    out_dir = Path(args.out_dir) / args.run / replay_dir.parents[3].name
    csv_path = out_dir / "latest_replay_residual_delta.csv"
    plot_path = out_dir / "latest_replay_residual_delta.png"
    meta_path = out_dir / "latest_replay_residual_delta.json"
    _save_csv(flat, csv_path)
    _save_plot(
        flat,
        plot_path,
        title=f"{args.run}: latest {args.num_transitions} replay transitions",
    )
    summary = {
        "run": args.run,
        "replay_dir": str(replay_dir),
        "num_transitions": args.num_transitions,
        "num_control_steps": len(flat),
        "actor_switch_rate": float(np.mean([r["actor_switch"] for r in flat])),
        "actual_delta_l2_mean": float(np.mean([r["actual_delta_l2"] for r in flat])),
        "actual_delta_l2_max": float(np.max([r["actual_delta_l2"] for r in flat])),
        "student_delta_l2_mean": float(np.mean([r["student_delta_l2"] for r in flat])),
        "student_delta_l2_max": float(np.max([r["student_delta_l2"] for r in flat])),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "source_files": [row["path"] for row in rows],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
