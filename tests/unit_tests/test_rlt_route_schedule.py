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

from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute


def _route_context(version: int = 0) -> RLTRouteContext:
    ref_chunk = torch.full((2, 2, 3), 0.25)
    return RLTRouteContext(
        env_obs={},
        rlt_obs={
            "z_rl": torch.zeros(2, 4),
            "proprio": torch.zeros(2, 1),
            "ref_chunk": ref_chunk,
        },
        student_actions=torch.ones(2, 2, 3),
        result={"forward_inputs": {"ref_chunk": ref_chunk}},
        mode="train",
        rlt_switch_flags=torch.tensor([[False], [True]]),
        version=version,
    )


def test_simulator_route_records_base_policy_warmup_transitions():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=2,
        record_base_warmup=True,
    )

    output = route.route(_route_context(version=0))

    expected_actions = torch.full((2, 2, 3), 0.25)
    torch.testing.assert_close(output.actions, expected_actions)
    assert output.result["forward_inputs"]["record_transition"].tolist() == [
        [True],
        [True],
    ]
    assert output.result["forward_inputs"]["actor_switch"].tolist() == [
        [False],
        [False],
    ]


def test_simulator_route_keeps_critical_phase_recording_after_warmup():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=2,
        record_base_warmup=True,
    )

    output = route.route(_route_context(version=2))

    assert output.result["forward_inputs"]["record_transition"].tolist() == [
        [False],
        [True],
    ]
    assert output.result["forward_inputs"]["actor_switch"].tolist() == [
        [False],
        [True],
    ]


def test_simulator_route_progressive_explore_starts_from_base_policy():
    route = SimulatorRLTRoute(
        use_schedule=True,
        warmup_updates=0,
        prog_explore_steps=100,
    )
    ctx = _route_context(version=0)
    ctx.rlt_switch_flags = torch.ones(2, 1, dtype=torch.bool)

    output = route.route(ctx)

    expected_actions = torch.full((2, 2, 3), 0.25)
    torch.testing.assert_close(output.actions, expected_actions)
    assert output.result["forward_inputs"]["actor_switch"].tolist() == [
        [False],
        [False],
    ]
    assert output.result["forward_inputs"]["prog_explore_ratio"].tolist() == [
        [0.0],
        [0.0],
    ]
