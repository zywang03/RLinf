# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class RLTFailureSignalEpisode:
    z_rl: torch.Tensor
    failed: bool


def hide_and_seek_loss(
    logits: torch.Tensor,
    failed: torch.Tensor,
    mask: torch.Tensor,
    *,
    intra_weight: float = 1.0,
    smoothness_weight: float = 0.0,
    margin: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = logits.float()
    failed = failed.bool()
    mask = mask.bool()
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [batch, time], got {logits.shape}.")
    if failed.shape != (logits.shape[0],):
        raise ValueError("failed must have shape [batch].")
    if mask.shape != logits.shape:
        raise ValueError("mask must match logits shape.")

    valid_episode = mask.any(dim=1)
    masked_logits = logits.masked_fill(~mask, -torch.inf)
    episode_max = torch.where(
        valid_episode,
        masked_logits.max(dim=1).values,
        torch.zeros_like(logits[:, 0]),
    )
    pair_mask = (
        failed[:, None]
        & (~failed[None, :])
        & valid_episode[:, None]
        & valid_episode[None, :]
    )
    pair_loss = F.softplus(float(margin) - episode_max[:, None] + episode_max[None, :])
    inter = _masked_mean(pair_loss, pair_mask)

    intra_values = []
    for row in range(logits.shape[0]):
        intra_values.append(
            _single_intra_loss(logits[row], mask[row], bool(failed[row]), margin=margin)
        )
    intra_values = torch.stack(intra_values)
    intra = _masked_mean(intra_values, failed & valid_episode)

    deltas = logits[:, 1:] - logits[:, :-1]
    delta_mask = mask[:, 1:] & mask[:, :-1]
    smoothness = _masked_mean(torch.square(deltas), delta_mask)
    total = inter + float(intra_weight) * intra + float(smoothness_weight) * smoothness
    return total, {
        "inter_loss": inter.detach(),
        "intra_loss": intra.detach(),
        "smoothness_loss": smoothness.detach(),
    }


class RLTFailureSignal(nn.Module):
    """Online failure detector over RLT embeddings.

    Parameters are persistent buffers so the actor can update them with a
    separate convergence loop without letting the SAC optimizer train this head.
    They still live in the policy state_dict, so existing actor-to-rollout
    weight sync carries them automatically.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        max_centers: int = 256,
        score_threshold: float = 0.7,
        distance_threshold: float = 6.0,
        pre_window: int = 2,
        post_window: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_centers = int(max_centers)
        self.score_threshold = float(score_threshold)
        self.distance_threshold = float(distance_threshold)
        self.pre_window = int(pre_window)
        self.post_window = int(post_window)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        self.register_buffer("normalizer_mean", torch.zeros(self.input_dim))
        self.register_buffer("normalizer_std", torch.ones(self.input_dim))
        self.register_buffer("w1", _xavier(self.input_dim, self.hidden_dim, generator))
        self.register_buffer("b1", torch.zeros(self.hidden_dim))
        self.register_buffer("w2", _xavier(self.hidden_dim, 1, generator))
        self.register_buffer("b2", torch.zeros(1))
        self.register_buffer(
            "failure_centers", torch.zeros(self.max_centers, self.input_dim)
        )
        self.register_buffer("center_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer("train_generation", torch.zeros((), dtype=torch.long))

    def transform(self, z_rl: torch.Tensor) -> torch.Tensor:
        z = z_rl.float().reshape(z_rl.shape[0], -1)
        return (z - self.normalizer_mean.float()) / (
            self.normalizer_std.float().clamp_min(1e-6)
        )

    def logits_from_transformed(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(x @ self.w1.float() + self.b1.float())
        return (hidden @ self.w2.float() + self.b2.float()).squeeze(-1)

    def scores_from_transformed(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits_from_transformed(x))

    def predict(self, z_rl: torch.Tensor) -> dict[str, torch.Tensor]:
        if z_rl.numel() == 0:
            empty = torch.zeros((0, 1), dtype=torch.bool, device=z_rl.device)
            return {"actor_switch": empty, "score": empty.float()}
        x = self.transform(z_rl)
        score = self.scores_from_transformed(x)
        score_gate = score >= self.score_threshold
        distance = torch.full_like(score, torch.inf)
        center_count = int(self.center_count.detach().cpu().item())
        if center_count > 0:
            centers = self.failure_centers[:center_count].to(device=x.device).float()
            distance = torch.cdist(x, centers).min(dim=1).values
        distance_gate = distance <= self.distance_threshold
        if bool(self.ready.detach().cpu().item()):
            gate = score_gate | distance_gate
        else:
            gate = torch.zeros_like(score_gate, dtype=torch.bool)
        return {
            "actor_switch": gate[:, None],
            "score": score[:, None],
            "distance": distance[:, None],
            "score_gate": score_gate[:, None],
            "distance_gate": distance_gate[:, None],
            "ready": torch.full(
                (z_rl.shape[0], 1),
                bool(self.ready.detach().cpu().item()),
                dtype=torch.bool,
                device=z_rl.device,
            ),
        }

    @torch.no_grad()
    def load_trained_state(
        self,
        *,
        mean: torch.Tensor,
        std: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        centers: torch.Tensor,
    ) -> None:
        device = self.normalizer_mean.device
        self.normalizer_mean.copy_(
            mean.to(device=device, dtype=self.normalizer_mean.dtype)
        )
        self.normalizer_std.copy_(
            std.to(device=device, dtype=self.normalizer_std.dtype)
        )
        self.w1.copy_(w1.to(device=device, dtype=self.w1.dtype))
        self.b1.copy_(b1.to(device=device, dtype=self.b1.dtype))
        self.w2.copy_(w2.to(device=device, dtype=self.w2.dtype))
        self.b2.copy_(b2.to(device=device, dtype=self.b2.dtype))
        self.failure_centers.zero_()
        count = min(int(centers.shape[0]), self.max_centers)
        if count > 0:
            self.failure_centers[:count].copy_(
                centers[:count].to(device=device, dtype=self.failure_centers.dtype)
            )
        self.center_count.fill_(count)
        self.ready.fill_(count > 0)
        self.train_generation.add_(1)


class RLTFailureSignalTrainer:
    def __init__(
        self,
        module: RLTFailureSignal,
        cfg: Any,
        *,
        device: torch.device,
    ) -> None:
        self.module = module
        self.cfg = cfg
        self.device = device
        self.episodes: list[RLTFailureSignalEpisode] = []
        self.failure_episode_count = 0
        self.trained_failure_episode_count = 0
        self.train_once_enabled = bool(
            cfg.get("train_mode", "warmup_once") == "warmup_once"
        )
        self.trained_once = False

    def state_dict(self) -> dict[str, Any]:
        save_episodes = not (self.train_once_enabled and self.trained_once)
        return {
            "episodes": [
                {
                    "z_rl": episode.z_rl.detach().float().cpu(),
                    "failed": bool(episode.failed),
                }
                for episode in self.episodes
            ]
            if save_episodes
            else [],
            "failure_episode_count": int(self.failure_episode_count),
            "trained_failure_episode_count": int(
                self.trained_failure_episode_count
            ),
            "trained_once": bool(self.trained_once),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.episodes = [
            RLTFailureSignalEpisode(
                z_rl=torch.as_tensor(episode["z_rl"]).float().cpu(),
                failed=bool(episode["failed"]),
            )
            for episode in state.get("episodes", [])
        ]
        self.failure_episode_count = int(state.get("failure_episode_count", 0))
        self.trained_failure_episode_count = int(
            state.get("trained_failure_episode_count", 0)
        )
        self.trained_once = bool(state.get("trained_once", False))

    def add_episodes(self, episodes: list[RLTFailureSignalEpisode]) -> int:
        if self.train_once_enabled and self.trained_once:
            return 0
        new_failures = 0
        if self.train_once_enabled:
            max_episodes = int(
                self.cfg.get("max_warmup_episodes", self.cfg.get("max_episodes", 512))
            )
        else:
            max_episodes = int(self.cfg.get("max_episodes", 512))
        for episode in episodes:
            z_rl = episode.z_rl.detach().float().cpu()
            if z_rl.numel() == 0:
                continue
            failed = bool(episode.failed)
            self.episodes.append(RLTFailureSignalEpisode(z_rl=z_rl, failed=failed))
            if failed:
                self.failure_episode_count += 1
                new_failures += 1
        if max_episodes > 0 and len(self.episodes) > max_episodes:
            dropped = self.episodes[: len(self.episodes) - max_episodes]
            self.failure_episode_count -= sum(int(ep.failed) for ep in dropped)
            self.trained_failure_episode_count = min(
                self.trained_failure_episode_count,
                self.failure_episode_count,
            )
            self.episodes = self.episodes[-max_episodes:]
        return new_failures

    def train_once(self) -> dict[str, float]:
        """Train at most once (used by warmup-only mode), then freeze."""
        if self.trained_once:
            return {
                "failure_signal/trained": 0.0,
                "failure_signal/episodes": float(len(self.episodes)),
                "failure_signal/failure_episodes": float(self.failure_episode_count),
                "failure_signal/trained_once": 0.0,
            }
        if not self.should_train():
            self.trained_once = True
            return {
                "failure_signal/trained": 0.0,
                "failure_signal/episodes": float(len(self.episodes)),
                "failure_signal/failure_episodes": float(self.failure_episode_count),
                "failure_signal/trained_once": 0.0,
            }
        metrics = self.train_to_convergence()
        self.trained_once = True
        metrics["failure_signal/trained_once"] = 1.0
        return metrics

    def should_train(self) -> bool:
        min_failures = int(self.cfg.get("min_failure_episodes", 1))
        min_successes = int(self.cfg.get("min_success_episodes", 0))
        success_count = sum(int(not ep.failed) for ep in self.episodes)
        return (
            self.failure_episode_count > self.trained_failure_episode_count
            and self.failure_episode_count >= min_failures
            and success_count >= min_successes
        )

    def train_to_convergence(self) -> dict[str, float]:
        if not self.should_train():
            return {
                "failure_signal/trained": 0.0,
                "failure_signal/episodes": float(len(self.episodes)),
                "failure_signal/failure_episodes": float(self.failure_episode_count),
            }

        embeddings, failed, mask = _pad_episodes(self.episodes, device=self.device)
        mean, std = _normalizer(embeddings, mask)
        x = (embeddings - mean) / std.clamp_min(1e-6)

        w1 = (
            self.module.w1.detach().float().to(self.device).clone().requires_grad_(True)
        )
        b1 = (
            self.module.b1.detach().float().to(self.device).clone().requires_grad_(True)
        )
        w2 = (
            self.module.w2.detach().float().to(self.device).clone().requires_grad_(True)
        )
        b2 = (
            self.module.b2.detach().float().to(self.device).clone().requires_grad_(True)
        )
        optimizer = torch.optim.Adam(
            [w1, b1, w2, b2],
            lr=float(self.cfg.get("lr", 3e-3)),
        )

        max_steps = int(self.cfg.get("max_steps", 500))
        min_steps = int(self.cfg.get("min_steps", 50))
        patience = int(self.cfg.get("patience", 25))
        tolerance = float(self.cfg.get("tolerance", 1e-4))
        best_loss = float("inf")
        stale_steps = 0
        last_metrics: dict[str, torch.Tensor] = {}
        steps_run = 0
        for step in range(max_steps):
            logits = _mlp_logits(x, w1, b1, w2, b2)
            loss, metrics = hide_and_seek_loss(
                logits,
                failed,
                mask,
                intra_weight=float(self.cfg.get("intra_weight", 1.0)),
                smoothness_weight=float(self.cfg.get("smoothness_weight", 0.0)),
                margin=float(self.cfg.get("margin", 1.0)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            steps_run = step + 1
            current = float(loss.detach().cpu().item())
            last_metrics = metrics
            if current < best_loss - tolerance:
                best_loss = current
                stale_steps = 0
            else:
                stale_steps += 1
            if steps_run >= min_steps and stale_steps >= patience:
                break

        with torch.no_grad():
            logits = _mlp_logits(x, w1, b1, w2, b2)
            scores = torch.sigmoid(logits)
            centers = _failure_centers(
                x,
                scores,
                failed,
                mask,
                max_centers=self.module.max_centers,
                pre=int(self.cfg.get("pre_window", self.module.pre_window)),
                post=int(self.cfg.get("post_window", self.module.post_window)),
            )
            self.module.load_trained_state(
                mean=mean.detach().cpu(),
                std=std.detach().cpu(),
                w1=w1.detach().cpu(),
                b1=b1.detach().cpu(),
                w2=w2.detach().cpu(),
                b2=b2.detach().cpu(),
                centers=centers.detach().cpu(),
            )

        self.trained_failure_episode_count = self.failure_episode_count
        return {
            "failure_signal/trained": 1.0,
            "failure_signal/episodes": float(len(self.episodes)),
            "failure_signal/failure_episodes": float(self.failure_episode_count),
            "failure_signal/success_episodes": float(
                sum(int(not ep.failed) for ep in self.episodes)
            ),
            "failure_signal/train_steps": float(steps_run),
            "failure_signal/loss": float(best_loss),
            "failure_signal/inter_loss": float(
                last_metrics.get("inter_loss", torch.tensor(0.0)).cpu().item()
            ),
            "failure_signal/intra_loss": float(
                last_metrics.get("intra_loss", torch.tensor(0.0)).cpu().item()
            ),
            "failure_signal/smoothness_loss": float(
                last_metrics.get("smoothness_loss", torch.tensor(0.0)).cpu().item()
            ),
            "failure_signal/center_count": float(
                self.module.center_count.detach().cpu().item()
            ),
            "failure_signal/generation": float(
                self.module.train_generation.detach().cpu().item()
            ),
        }


def _xavier(input_dim: int, output_dim: int, generator: torch.Generator) -> torch.Tensor:
    limit = (6.0 / float(input_dim + output_dim)) ** 0.5
    return torch.empty(input_dim, output_dim).uniform_(
        -limit,
        limit,
        generator=generator,
    )


def _mlp_logits(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    hidden = torch.tanh(flat @ w1 + b1)
    logits = (hidden @ w2 + b2).reshape(*x.shape[:2])
    return logits


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / denom


def _single_intra_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    failed: bool,
    *,
    margin: float,
) -> torch.Tensor:
    if not failed:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    adjacent = mask[1:] & mask[:-1]
    if not adjacent.any():
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    deltas = logits[1:] - logits[:-1]
    onset_scores = torch.where(adjacent, deltas, torch.full_like(deltas, -torch.inf))
    onset = int(onset_scores.argmax().item()) + 1
    time = torch.arange(logits.shape[0], device=logits.device)
    pre_mask = mask & (time < onset)
    post_mask = mask & (time >= onset)
    if not pre_mask.any() or not post_mask.any():
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    pre = _masked_mean(logits, pre_mask)
    post = _masked_mean(logits, post_mask)
    return F.softplus(float(margin) - post + pre)


def _pad_episodes(
    episodes: list[RLTFailureSignalEpisode],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(ep.z_rl.shape[0]) for ep in episodes)
    input_dim = int(episodes[0].z_rl.reshape(episodes[0].z_rl.shape[0], -1).shape[-1])
    embeddings = torch.zeros(len(episodes), max_len, input_dim, device=device)
    mask = torch.zeros(len(episodes), max_len, dtype=torch.bool, device=device)
    failed = torch.zeros(len(episodes), dtype=torch.bool, device=device)
    for row, episode in enumerate(episodes):
        z = episode.z_rl.reshape(episode.z_rl.shape[0], -1).to(device=device).float()
        embeddings[row, : z.shape[0]] = z
        mask[row, : z.shape[0]] = True
        failed[row] = bool(episode.failed)
    return embeddings, failed, mask


def _normalizer(
    embeddings: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(dtype=embeddings.dtype).unsqueeze(-1)
    denom = weights.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (embeddings * weights).sum(dim=(0, 1)) / denom
    var = ((embeddings - mean).square() * weights).sum(dim=(0, 1)) / denom
    return mean, var.clamp_min(1e-6).sqrt()


def _failure_centers(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    failed: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_centers: int,
    pre: int,
    post: int,
) -> torch.Tensor:
    centers = []
    for row in torch.nonzero(failed, as_tuple=False).reshape(-1).tolist():
        valid_idx = torch.nonzero(mask[row], as_tuple=False).reshape(-1)
        if valid_idx.numel() == 0:
            continue
        valid_scores = scores[row, valid_idx]
        if valid_idx.numel() == 1:
            onset_pos = 0
        else:
            deltas = valid_scores[1:] - valid_scores[:-1]
            onset_pos = int(deltas.argmax().item()) + 1
        start = max(0, onset_pos - max(0, int(pre)))
        stop = min(int(valid_idx.numel()), onset_pos + max(0, int(post)) + 1)
        centers.append(embeddings[row, valid_idx[start:stop]])
    if not centers:
        return torch.zeros(0, embeddings.shape[-1], device=embeddings.device)
    values = torch.cat(centers, dim=0)
    if values.shape[0] <= max_centers:
        return values
    select = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=max_centers,
        device=values.device,
    ).round().long()
    return values.index_select(0, select)
