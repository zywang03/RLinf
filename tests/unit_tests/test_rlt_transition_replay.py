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

from types import SimpleNamespace

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACReplayMixin


class _FakeReplayBuffer:
    _flatten_trajectory = TrajectoryReplayBuffer._flatten_trajectory


class _FakeTrain(dict):
    pass


def _obs(steps: int) -> dict[str, torch.Tensor]:
    return {
        "z_rl": torch.randn(steps, 2, 3),
        "proprio": torch.randn(steps, 2, 2),
        "ref_chunk": torch.randn(steps, 2, 2, 4),
    }


def _worker() -> RLTACReplayMixin:
    worker = object.__new__(RLTACReplayMixin)
    worker.replay_buffer = _FakeReplayBuffer()
    worker.cfg = SimpleNamespace(
        env=SimpleNamespace(train=_FakeTrain(auto_reset=False))
    )
    return worker


def test_rlt_transition_replay_skips_final_bootstrap_row_without_curr_obs():
    record_transition = torch.zeros(11, 2, dtype=torch.bool)
    record_transition[0, 0] = True
    record_transition[10, 1] = True

    trajectory = Trajectory(
        max_episode_length=1,
        actions=torch.randn(11, 2, 4),
        rewards=torch.randn(11, 2),
        dones=torch.zeros(12, 2, dtype=torch.bool),
        forward_inputs={"record_transition": record_transition},
        curr_obs=_obs(10),
        next_obs=_obs(10),
    )

    replayed, completed = _worker()._transition_replay_trajectories(trajectory)

    assert completed == 0
    assert len(replayed) == 1
    assert replayed[0].actions.shape == (1, 1, 4)
    assert replayed[0].curr_obs["z_rl"].shape == (1, 1, 3)
