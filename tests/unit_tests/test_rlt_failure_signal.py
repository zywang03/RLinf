import torch

from rlinf.algorithms.rlt.failure_signal import (
    RLTFailureSignal,
    RLTFailureSignalEpisode,
    RLTFailureSignalTrainer,
)
from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import RealworldRLTRoute


def test_failure_signal_trains_on_new_failure_and_gates_near_onset():
    module = RLTFailureSignal(
        input_dim=2,
        hidden_dim=8,
        max_centers=8,
        distance_threshold=0.5,
        score_threshold=2.0,
    )
    trainer = RLTFailureSignalTrainer(
        module,
        {
            "min_failure_episodes": 1,
            "min_success_episodes": 1,
            "min_steps": 5,
            "max_steps": 60,
            "patience": 5,
            "lr": 0.02,
            "pre_window": 0,
            "post_window": 0,
        },
        device=torch.device("cpu"),
    )
    episodes = [
        RLTFailureSignalEpisode(
            z_rl=torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0]]),
            failed=True,
        ),
        RLTFailureSignalEpisode(
            z_rl=torch.tensor([[0.0, 0.1], [0.1, 0.2], [0.2, 0.2]]),
            failed=False,
        ),
    ]

    assert trainer.add_episodes(episodes) == 1
    metrics = trainer.train_to_convergence()
    output = module.predict(torch.tensor([[3.0, 3.0], [0.0, 0.0]]))

    assert metrics["failure_signal/trained"] == 1.0
    assert metrics["failure_signal/center_count"] > 0
    assert output["actor_switch"][0].item()
    assert not output["actor_switch"][1].item()


def test_failure_signal_warmup_once_trains_once_and_freezes():
    module = RLTFailureSignal(
        input_dim=2,
        hidden_dim=8,
        max_centers=8,
        distance_threshold=0.5,
        score_threshold=2.0,
    )
    trainer = RLTFailureSignalTrainer(
        module,
        {
            "train_mode": "warmup_once",
            "max_warmup_episodes": 0,
            "min_failure_episodes": 1,
            "min_success_episodes": 0,
            "min_steps": 5,
            "max_steps": 60,
            "patience": 5,
            "lr": 0.02,
            "pre_window": 0,
            "post_window": 0,
        },
        device=torch.device("cpu"),
    )
    episodes = [
        RLTFailureSignalEpisode(
            z_rl=torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0]]),
            failed=True,
        ),
        RLTFailureSignalEpisode(
            z_rl=torch.tensor([[0.0, 0.1], [0.1, 0.2], [0.2, 0.2]]),
            failed=False,
        ),
    ]

    assert trainer.add_episodes(episodes) == 1
    metrics = trainer.train_once()

    assert metrics["failure_signal/trained"] == 1.0
    assert metrics["failure_signal/trained_once"] == 1.0
    assert trainer.trained_once
    assert bool(module.ready.detach().cpu().item())

    # After the one-shot training, new episodes are rejected and no retrain
    # happens, so the detector stays frozen at the warmup-trained state.
    extra = [
        RLTFailureSignalEpisode(
            z_rl=torch.tensor([[5.0, 5.0], [6.0, 6.0]]),
            failed=True,
        )
    ]
    assert trainer.add_episodes(extra) == 0
    assert len(trainer.episodes) == 2
    metrics_again = trainer.train_once()
    assert metrics_again["failure_signal/trained"] == 0.0
    assert metrics_again["failure_signal/trained_once"] == 0.0
    assert trainer.module.train_generation.detach().cpu().item() == 1


def test_failure_signal_train_mode_defaults_to_warmup_once():
    module = RLTFailureSignal(
        input_dim=2,
        hidden_dim=8,
        max_centers=8,
        distance_threshold=0.5,
        score_threshold=2.0,
    )
    trainer = RLTFailureSignalTrainer(module, {}, device=torch.device("cpu"))
    assert trainer.train_once_enabled


def test_failure_signal_trainer_state_preserves_frozen_warmup_once():
    module = RLTFailureSignal(
        input_dim=2,
        hidden_dim=8,
        max_centers=8,
        distance_threshold=0.5,
        score_threshold=2.0,
    )
    cfg = {
        "train_mode": "warmup_once",
        "max_warmup_episodes": 0,
        "min_failure_episodes": 1,
        "min_success_episodes": 0,
        "min_steps": 5,
        "max_steps": 60,
        "patience": 5,
        "lr": 0.02,
        "pre_window": 0,
        "post_window": 0,
    }
    trainer = RLTFailureSignalTrainer(
        module,
        cfg,
        device=torch.device("cpu"),
    )
    trainer.add_episodes(
        [
            RLTFailureSignalEpisode(
                z_rl=torch.tensor([[0.0, 0.0], [3.0, 3.0]]),
                failed=True,
            ),
            RLTFailureSignalEpisode(
                z_rl=torch.tensor([[0.0, 0.1], [0.2, 0.2]]),
                failed=False,
            ),
        ]
    )
    trainer.train_once()

    state = trainer.state_dict()
    assert state["trained_once"] is True
    assert state["episodes"] == []
    assert state["failure_episode_count"] == 1
    assert state["trained_failure_episode_count"] == 1

    restored = RLTFailureSignalTrainer(
        module,
        cfg,
        device=torch.device("cpu"),
    )
    restored.load_state_dict(state)
    assert restored.trained_once is True
    assert restored.add_episodes(
        [
            RLTFailureSignalEpisode(
                z_rl=torch.tensor([[5.0, 5.0]]),
                failed=True,
            )
        ]
    ) == 0


def test_predict_rlt_actions_or_merges_failure_gate_with_existing_switch():
    policy = _FakePolicy()
    feature_model = _FakeFeatureModel()
    route = RealworldRLTRoute()
    env_obs = {"states": torch.zeros(2, 1)}
    rlt_switch_flags = torch.tensor([[False], [True]])

    actions, result = predict_rlt_actions(
        policy_model=policy,
        feature_model=feature_model,
        rlt_route=route,
        env_obs=env_obs,
        final_obs=None,
        mode="train",
        rlt_switch_flags=rlt_switch_flags,
    )

    assert torch.equal(actions, torch.ones(2, 2, 3))
    assert result["forward_inputs"]["rlt_failure_signal_actor_switch"].tolist() == [
        [True],
        [False],
    ]
    assert result["forward_inputs"]["actor_switch"].tolist() == [[True], [True]]


class _FakeFeatureModel:
    def extract_rlt_obs(self, env_obs):
        del env_obs
        return {
            "z_rl": torch.zeros(2, 4),
            "proprio": torch.zeros(2, 1),
            "ref_chunk": torch.zeros(2, 2, 3),
        }


class _FakePolicy:
    def predict_action_batch(self, env_obs, mode, return_obs):
        del mode
        actions = torch.ones(2, 2, 3)
        forward_inputs = {"action": actions.reshape(2, -1)}
        if return_obs:
            forward_inputs.update(env_obs)
        return actions, {
            "prev_logprobs": torch.zeros(2, 6),
            "prev_values": torch.zeros(2, 1),
            "forward_inputs": forward_inputs,
        }

    def predict_failure_signal_gate(self, rlt_obs):
        del rlt_obs
        return {
            "actor_switch": torch.tensor([[True], [False]]),
            "score": torch.tensor([[0.9], [0.1]]),
        }
