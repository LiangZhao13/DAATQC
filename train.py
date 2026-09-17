"""Initial DAATQC training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from daatqc.config import DisturbanceConfig, TrainingConfig
from daatqc.training import build_model, make_callbacks, make_vector_environments


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DAATQC from scratch")
    parser.add_argument("--timesteps", type=int, default=1_200_000)
    parser.add_argument("--output-dir", default="./drl_dp_DAATQC_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--current-speed", type=float, default=0.0)
    parser.add_argument("--wind-speed", type=float, default=0.0)
    parser.add_argument("--wave-height", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    disturbance = DisturbanceConfig(
        current_speed=arguments.current_speed,
        wind_speed_mean=arguments.wind_speed,
        significant_wave_height=arguments.wave_height,
    )
    config = TrainingConfig(
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        number_of_environments=arguments.n_envs,
        total_timesteps=arguments.timesteps,
        device=arguments.device,
        disturbance=disturbance,
    )
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    train_environment, evaluation_environment = make_vector_environments(config)
    model = build_model(config, train_environment)
    callbacks = make_callbacks(
        config,
        evaluation_environment,
        "training_log.csv",
        "best_model",
        "eval_log",
        "checkpoints",
        "DAATQC",
    )
    try:
        model.learn(config.total_timesteps, callback=callbacks, progress_bar=False)
        model.save(str(Path(config.output_dir) / "DAATQC_final"))
    finally:
        train_environment.close()
        evaluation_environment.close()
    print(f"Training complete: {Path(config.output_dir) / 'DAATQC_final.zip'}")


if __name__ == "__main__":
    main()
