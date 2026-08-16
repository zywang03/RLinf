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

import torch

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _obs(batch_size=3):
    return {
        "z_rl": torch.randn(batch_size, 5),
        "proprio": torch.randn(batch_size, 2),
        "ref_chunk": torch.linspace(-0.5, 0.5, batch_size * 2 * 4).reshape(
            batch_size, 2, 4
        ),
    }


def test_rlt_residual_actor_is_zero_mean_at_initialization():
    model = RLTMLPPolicy(
        z_dim=5,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        residual_actor=True,
    )

    obs = _obs()
    action, _, _ = model.sac_forward(obs, deterministic=True)

    expected = obs["ref_chunk"].reshape(obs["ref_chunk"].shape[0], -1)
    torch.testing.assert_close(action, expected)


def test_rlt_policy_keeps_twin_q_heads():
    model = RLTMLPPolicy(
        z_dim=5,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        residual_actor=True,
    )
    obs = _obs()
    actions = torch.randn(3, 2, 4)

    q_values = model.sac_q_forward(obs, actions)

    assert q_values.shape == (3, 2)


def test_rlt_policy_default_builds_layernorm_critic():
    model = RLTMLPPolicy(
        z_dim=5,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        q_head_hidden_dim=16,
        q_head_num_blocks=2,
        residual_actor=True,
    )
    obs = _obs()
    actions = torch.randn(3, 2, 4)

    q_values = model.sac_q_forward(obs, actions)

    assert q_values.shape == (3, 2)
    assert all(
        len([module for module in qf.modules() if isinstance(module, torch.nn.LayerNorm)])
        >= 3
        for qf in model.q_head.qs
    )


def test_rlt_policy_supports_relu_actor_and_learned_std():
    model = RLTMLPPolicy(
        z_dim=5,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        actor_activation="relu",
        actor_std_type="learned",
        log_std_min=-20,
        init_learned_std_to_fixed=False,
        residual_actor=True,
    )
    obs = _obs()
    actions = torch.randn(3, 2, 4)

    q_values = model.sac_q_forward(obs, actions)

    assert q_values.shape == (3, 2)
    assert isinstance(model.backbone[1], torch.nn.ReLU)
    assert model.logstd_range == (-20.0, 2)


def test_rlt_residual_actor_keeps_zero_mean_with_learned_std():
    model = RLTMLPPolicy(
        z_dim=5,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        actor_std_type="learned",
        fixed_std=0.1,
        residual_actor=True,
    )
    obs = _obs()

    action, logprobs, _ = model.sac_forward(obs, deterministic=True)

    expected = obs["ref_chunk"].reshape(obs["ref_chunk"].shape[0], -1)
    torch.testing.assert_close(action, expected)
    assert logprobs.shape == action.shape


def test_rlt_residual_actor_squashes_residual_before_adding_ref():
    model = RLTMLPPolicy(
        z_dim=5,
        proprio_dim=2,
        action_dim=4,
        num_action_chunks=2,
        residual_actor=True,
        residual_scale=0.25,
    )
    obs = _obs()
    with torch.no_grad():
        model.actor_mean.bias.fill_(2.0)

    action, _, _ = model.sac_forward(obs, deterministic=True)

    ref = obs["ref_chunk"].reshape(obs["ref_chunk"].shape[0], -1)
    expected = ref + 0.25 * torch.tanh(torch.full_like(ref, 2.0))
    torch.testing.assert_close(action, expected)
