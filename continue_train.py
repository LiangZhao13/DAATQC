"""Progressive-curriculum continuation training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import TQC

from daatqc.config import DisturbanceConfig, SuccessCriteria, TrainingConfig
from daatqc.training import find_pretrained_model, make_callbacks, make_vector_environments


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue DAATQC training")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--output-dir", default="./drl_dp_DAATQC_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--position-threshold", type=float, default=0.20)
    parser.add_argument("--heading-threshold-deg", type=float, default=2.0)
    parser.add_argument("--speed-threshold", type=float, default=0.08)
    parser.add_argument("--hold-steps", type=int, default=50)
    parser.add_argument("--keep-log", action="store_true")
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
    success = SuccessCriteria(
        position_threshold=arguments.position_threshold,
        heading_threshold_deg=arguments.heading_threshold_deg,
        speed_threshold=arguments.speed_threshold,
        hold_steps=arguments.hold_steps,
    )
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    continued_log = output / "continued_training_log.csv"
    if not arguments.keep_log and continued_log.exists():
        continued_log.unlink()

    training_environment, evaluation_environment = make_vector_environments(
        config, success=success, seed_offset=10_000
    )
    model_path = find_pretrained_model(config.output_dir, arguments.model_path)
    model = TQC.load(
        model_path,
        env=training_environment,
        device=config.device,
        print_system_info=False,
    )
    callbacks = make_callbacks(
        config,
        evaluation_environment,
        "continued_training_log.csv",
        "continued_best_model",
        "continued_eval_log",
        "continued_checkpoints",
        "DAATQC_continue",
    )
    try:
        model.learn(
            config.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        model.save(str(output / "DAATQC_continued_final"))
    finally:
        training_environment.close()
        evaluation_environment.close()
    print(f"Continuation complete: {output / 'DAATQC_continued_final.zip'}")


if __name__ == "__main__":
    main()
