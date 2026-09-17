"""Evaluate a trained DAATQC policy on the 300 s four-corner test."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import TQC

from daatqc.config import DisturbanceConfig, EnvironmentConfig, SuccessCriteria
from daatqc.environment import DPDirectRLEnv


REFERENCE_SEQUENCE = (
    (0.0, 0.0, 0.0),
    (8.0, 0.0, 0.0),
    (8.0, 8.0, 0.0),
    (0.0, 8.0, 0.0),
    (0.0, 8.0, 30.0),
    (0.0, 0.0, 30.0),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DAATQC four-corner test")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="./evaluation_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--current-speed", type=float, default=0.0)
    parser.add_argument("--wind-speed", type=float, default=0.0)
    parser.add_argument("--wave-height", type=float, default=0.0)
    return parser.parse_args()


def reference_at_time(time_seconds: float) -> np.ndarray:
    index = min(int(time_seconds // 50.0), len(REFERENCE_SEQUENCE) - 1)
    north, east, heading_deg = REFERENCE_SEQUENCE[index]
    return np.array([north, east, np.deg2rad(heading_deg)], dtype=np.float64)


def main() -> None:
    arguments = parse_arguments()
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    disturbance = DisturbanceConfig(
        current_speed=arguments.current_speed,
        wind_speed_mean=arguments.wind_speed,
        significant_wave_height=arguments.wave_height,
    )
    config = EnvironmentConfig(
        time_step=0.1,
        episode_steps=3000,
        training=False,
        measurement_noise=False,
        disturbance=disturbance,
        success=SuccessCriteria(hold_steps=10**9),
    )
    environment = DPDirectRLEnv(config)
    observation, _ = environment.reset(
        seed=arguments.seed,
        options={"eta_ref": np.zeros(3), "eta0": np.zeros(3), "nu0": np.zeros(3), "n0": np.zeros(6)},
    )
    model = TQC.load(arguments.model_path, device=arguments.device)
    records = []
    for step in range(config.episode_steps):
        time_seconds = step * config.time_step
        environment.reference = reference_at_time(time_seconds)
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        records.append(
            {
                "time_s": time_seconds,
                "north_m": environment.eta[0],
                "east_m": environment.eta[1],
                "heading_deg": np.rad2deg(environment.eta[2]),
                "target_north_m": environment.reference[0],
                "target_east_m": environment.reference[1],
                "target_heading_deg": np.rad2deg(environment.reference[2]),
                "position_error_m": info["pos_err"],
                "heading_error_deg": info["yaw_err_deg"],
                "energy_cost": info["energy_cost"],
                "reward": reward,
            }
        )
        if terminated and not info.get("is_success", False):
            break
    environment.close()

    frame = pd.DataFrame(records)
    frame.to_csv(output / "four_corner_test.csv", index=False)
    figure, axis = plt.subplots(figsize=(7.5, 7.0))
    axis.plot(frame["east_m"], frame["north_m"], label="DAATQC", linewidth=2.0)
    targets = np.asarray([[entry[1], entry[0]] for entry in REFERENCE_SEQUENCE])
    axis.plot(targets[:, 0], targets[:, 1], "ko--", label="Reference", alpha=0.7)
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "four_corner_trajectory.svg", bbox_inches="tight")
    plt.close(figure)
    print(f"Saved evaluation outputs to {output}")


if __name__ == "__main__":
    main()

