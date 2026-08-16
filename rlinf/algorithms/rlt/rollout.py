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

from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = feature_model.extract_rlt_obs(final_obs)
    for key in RLT_OBS_KEYS:
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]


def _merge_optional_switch_flags(
    rlt_switch_flags: torch.Tensor | None,
    failure_switch: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if failure_switch is None:
        return rlt_switch_flags
    failure_switch = torch.as_tensor(failure_switch, device=device).bool()
    if failure_switch.dim() == 1:
        failure_switch = failure_switch[:, None]
    failure_switch = failure_switch.reshape(batch_size, -1)
    if rlt_switch_flags is None:
        return failure_switch
    rlt_switch_flags = torch.as_tensor(rlt_switch_flags, device=device).bool()
    if rlt_switch_flags.dim() == 1:
        rlt_switch_flags = rlt_switch_flags[:, None]
    rlt_switch_flags = rlt_switch_flags.reshape(batch_size, -1)
    if failure_switch.shape[1] == 1 and rlt_switch_flags.shape[1] > 1:
        failure_switch = failure_switch.expand(-1, rlt_switch_flags.shape[1])
    elif rlt_switch_flags.shape[1] == 1 and failure_switch.shape[1] > 1:
        rlt_switch_flags = rlt_switch_flags.expand(-1, failure_switch.shape[1])
    return rlt_switch_flags | failure_switch


def _append_failure_signal_outputs(
    result: dict[str, Any],
    outputs: dict[str, torch.Tensor] | None,
) -> None:
    if not outputs:
        return
    forward_inputs = result["forward_inputs"]
    key_map = {
        "actor_switch": "rlt_failure_signal_actor_switch",
        "score": "rlt_failure_signal_score",
        "distance": "rlt_failure_signal_distance",
        "score_gate": "rlt_failure_signal_score_gate",
        "distance_gate": "rlt_failure_signal_distance_gate",
        "ready": "rlt_failure_signal_ready",
    }
    for src_key, dst_key in key_map.items():
        value = outputs.get(src_key)
        if isinstance(value, torch.Tensor):
            forward_inputs[dst_key] = value.detach()


def predict_rlt_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    rlt_route: RLTRoute,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
    expert_model: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        rlt_obs = feature_model.extract_rlt_obs(env_obs)
        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        failure_outputs = None
        if hasattr(policy_model, "predict_failure_signal_gate"):
            failure_outputs = policy_model.predict_failure_signal_gate(rlt_obs)
        if failure_outputs:
            rlt_switch_flags = _merge_optional_switch_flags(
                rlt_switch_flags,
                failure_outputs.get("actor_switch"),
                batch_size=actions.shape[0],
                device=actions.device,
            )
            _append_failure_signal_outputs(result, failure_outputs)

        route_output = rlt_route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=rlt_obs,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=expert_model,
                version=version,
            )
        )
        actions = route_output.actions
        result = route_output.result

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
        )

    return actions, result
