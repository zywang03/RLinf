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

import math

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.algorithms.rlt.failure_signal import RLTFailureSignal
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


class RLTMLPPolicy(MLPPolicy):
    """MLP actor-critic policy for RLT Stage 2 heads.

    Actor input follows RLT: reference action chunk, RL token feature, and
    proprioceptive state. Critic input follows RLT: action chunk, RL token
    feature, and proprioceptive state.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        q_head_hidden_dim: int | None = None,
        q_head_num_blocks: int | None = None,
        fixed_std: float = 0.002,
        actor_std_type: str = "fixed",
        use_tanh_logprob_correction: bool | None = None,
        residual_actor: bool = False,
        residual_scale: float = 1.0,
        failure_signal: dict | None = None,
        actor_activation: str = "tanh",
        log_std_min: float = -5,
        log_std_max: float = 2.0,
        init_learned_std_to_fixed: bool = True,
    ):
        if not add_q_head:
            raise ValueError(
                "RLTMLPPolicy requires add_q_head=True for actor-critic training."
            )
        z_dim = int(z_dim)
        proprio_dim = int(proprio_dim)
        step_action_dim = int(action_dim)
        chunk_len = int(num_action_chunks)
        ref_chunk_len = (
            chunk_len if ref_num_action_chunks is None else int(ref_num_action_chunks)
        )
        if ref_chunk_len < chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{ref_chunk_len} < {chunk_len}."
            )
        flat_action_dim = chunk_len * step_action_dim

        actor_obs_dim = z_dim + proprio_dim + flat_action_dim
        critic_obs_dim = z_dim + proprio_dim

        resolved_q_head_type = (
            "layernorm" if str(q_head_type) == "default" else str(q_head_type)
        )

        super().__init__(
            obs_dim=actor_obs_dim,
            action_dim=flat_action_dim,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=add_q_head,
            q_head_type=resolved_q_head_type,
            critic_obs_dim=critic_obs_dim,
            q_head_hidden_dim=q_head_hidden_dim,
            q_head_num_blocks=q_head_num_blocks,
            actor_activation=actor_activation,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.step_action_dim = step_action_dim
        self.chunk_len = chunk_len
        self.ref_chunk_len = ref_chunk_len
        self.flat_action_dim = flat_action_dim
        self.actor_std_type = str(actor_std_type)
        if self.actor_std_type not in {"fixed", "learned"}:
            raise ValueError(
                "actor_std_type must be either 'fixed' or 'learned', got "
                f"{self.actor_std_type!r}."
            )
        self.fixed_std = float(fixed_std)
        if self.actor_std_type == "fixed" and self.fixed_std <= 0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")
        if use_tanh_logprob_correction is None:
            use_tanh_logprob_correction = self.actor_std_type == "learned"
        self.use_tanh_logprob_correction = bool(use_tanh_logprob_correction)
        self.init_learned_std_to_fixed = bool(init_learned_std_to_fixed)
        if self.actor_std_type == "learned" and self.init_learned_std_to_fixed:
            self._init_learned_std_head()
        self.residual_actor = bool(residual_actor)
        self.residual_scale = float(residual_scale)
        if self.residual_scale < 0:
            raise ValueError(
                f"residual_scale must be non-negative, got {self.residual_scale}."
            )
        if self.residual_actor:
            torch.nn.init.zeros_(self.actor_mean.weight)
            torch.nn.init.zeros_(self.actor_mean.bias)
        self.rlt_failure_signal = self._build_failure_signal(failure_signal)

    def _build_failure_signal(self, cfg: dict | None) -> RLTFailureSignal | None:
        if cfg is None or not bool(cfg.get("enable", False)):
            return None
        return RLTFailureSignal(
            self.z_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            max_centers=int(cfg.get("max_centers", 256)),
            score_threshold=float(cfg.get("score_threshold", 0.7)),
            distance_threshold=float(cfg.get("distance_threshold", 6.0)),
            pre_window=int(cfg.get("pre_window", 2)),
            post_window=int(cfg.get("post_window", 1)),
            seed=int(cfg.get("seed", 0)),
        )

    def _init_learned_std_head(self) -> None:
        target_logstd = min(
            max(math.log(max(self.fixed_std, 1e-8)), self.logstd_range[0]),
            self.logstd_range[1],
        )
        normalized = 2 * (target_logstd - self.logstd_range[0]) / (
            self.logstd_range[1] - self.logstd_range[0]
        ) - 1
        bias = math.atanh(max(min(normalized, 0.999), -0.999))
        torch.nn.init.zeros_(self.actor_logstd.weight)
        torch.nn.init.constant_(self.actor_logstd.bias, bias)

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed = {}
        for key, value in env_obs.items():
            processed[key] = value.to(device) if torch.is_tensor(value) else value
        return processed

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _get_z(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["z_rl"])

    def _get_proprio(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["proprio"])

    def _get_ref_chunk(self, obs: dict) -> torch.Tensor:
        ref_chunk = self._flatten_batch(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, self.step_action_dim
        )
        ref_chunk = ref_chunk[:, : self.chunk_len]
        return ref_chunk.reshape(ref_chunk.shape[0], -1)

    def _maybe_drop_reference(
        self,
        ref_chunk: torch.Tensor,
        reference_dropout_prob: float,
    ) -> torch.Tensor:
        if reference_dropout_prob <= 0:
            return ref_chunk
        keep_prob = 1.0 - float(reference_dropout_prob)
        keep_mask = (
            torch.rand((ref_chunk.shape[0], 1), device=ref_chunk.device) < keep_prob
        )
        return ref_chunk * keep_mask.to(dtype=ref_chunk.dtype)

    def _actor_state(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        actor_state, _ = self._actor_state_and_ref(
            obs,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        )
        return actor_state

    def _actor_state_and_ref(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ref_chunk = self._get_ref_chunk(obs)
        actor_ref_chunk = ref_chunk
        if apply_reference_dropout:
            actor_ref_chunk = self._maybe_drop_reference(
                ref_chunk, reference_dropout_prob
            )
        actor_state = torch.cat(
            [actor_ref_chunk, self._get_z(obs), self._get_proprio(obs)], dim=-1
        )
        return actor_state, ref_chunk

    def _apply_residual_actor(
        self,
        residual_action: torch.Tensor,
        ref_chunk: torch.Tensor,
    ) -> torch.Tensor:
        if not self.residual_actor:
            return residual_action
        return ref_chunk + self.residual_scale * residual_action

    def _actor_std(
        self, feat: torch.Tensor, action_mean: torch.Tensor
    ) -> torch.Tensor:
        if self.actor_std_type == "fixed":
            return torch.full_like(action_mean, self.fixed_std)
        action_logstd = self.actor_logstd(feat)
        action_logstd = torch.tanh(action_logstd)
        action_logstd = self.logstd_range[0] + 0.5 * (
            self.logstd_range[1] - self.logstd_range[0]
        ) * (action_logstd + 1)
        return torch.exp(action_logstd)

    def _critic_state(self, obs: dict) -> torch.Tensor:
        return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def sac_forward(
        self,
        obs,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        deterministic: bool = False,
        **kwargs,
    ):
        actor_state, ref_chunk = self._actor_state_and_ref(
            obs,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        )
        feat = self.backbone(actor_state)
        action_mean = self.actor_mean(feat)
        action_std = self._actor_std(feat, action_mean)
        probs = Normal(action_mean, action_std)
        raw_action = action_mean if deterministic else probs.rsample()
        chunk_logprobs = probs.log_prob(raw_action)
        squashed_action = torch.tanh(raw_action)
        if self.use_tanh_logprob_correction:
            chunk_logprobs = chunk_logprobs - torch.log(
                1 - squashed_action.pow(2) + 1e-6
            )
        action = self._apply_residual_actor(squashed_action, ref_chunk)
        return action, chunk_logprobs, None

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        del shared_feature
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        return self.q_head(critic_state, self._flatten_batch(actions))

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        del shared_feature
        critic_state = self._critic_state(obs)
        next_critic_state = (
            self._critic_state(next_obs) if next_obs is not None else None
        )
        if detach_encoder:
            critic_state = critic_state.detach()
            if next_critic_state is not None:
                next_critic_state = next_critic_state.detach()
        return self.q_head(
            critic_state,
            self._flatten_batch(actions),
            next_state_features=next_critic_state,
            next_action_features=(
                self._flatten_batch(next_actions) if next_actions is not None else None
            ),
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def sft_forward(self, data, **kwargs):
        obs = data["obs"] if "obs" in data else data
        target_actions = self._flatten_batch(
            data["action"] if "action" in data else data["actions"]
        )
        actor_state, ref_chunk = self._actor_state_and_ref(obs)
        pred_actions = self.actor_mean(self.backbone(actor_state))
        if self.residual_actor:
            pred_actions = torch.tanh(pred_actions)
        pred_actions = self._apply_residual_actor(pred_actions, ref_chunk)
        return F.mse_loss(pred_actions, target_actions, reduction="none")

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values
        obs = self.preprocess_env_obs(env_obs=env_obs)
        action, chunk_logprobs, _ = self.sac_forward(
            obs, deterministic=(mode == "eval")
        )
        chunk_actions = self._format_chunk_actions(action)

        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(obs)

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": torch.zeros_like(chunk_logprobs[..., :1]),
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    @torch.inference_mode()
    def predict_failure_signal_gate(self, env_obs: dict) -> dict[str, torch.Tensor] | None:
        if self.rlt_failure_signal is None:
            return None
        obs = self.preprocess_env_obs(env_obs=env_obs)
        return self.rlt_failure_signal.predict(obs["z_rl"])
