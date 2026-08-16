#!/usr/bin/env python
import argparse
import faulthandler
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--shader-pack", default="minimal")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()

    faulthandler.dump_traceback_later(
        args.timeout_seconds,
        repeat=False,
        file=sys.stderr,
    )

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        num_envs=args.num_envs,
        sim_backend="gpu",
        sensor_configs={
            "shader_pack": args.shader_pack,
            "width": args.width,
            "height": args.height,
        },
    )
    print(f"OK: created {args.env_id} with num_envs={env.unwrapped.num_envs}")
    sensors = getattr(env.unwrapped, "_sensors", {})
    for name, sensor in sensors.items():
        config = getattr(sensor, "config", None)
        if config is not None:
            print(
                f"sensor {name}: {config.width}x{config.height}, "
                f"shader={config.shader_pack}"
            )
    env.close()


if __name__ == "__main__":
    main()
