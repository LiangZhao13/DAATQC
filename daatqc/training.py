"""Shared model, environment, and callback factories."""

from __future__ import annotations

import os
from pathlib import Path

from sb3_contrib import TQC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .attention import DemandActuatorAttentionExtractor
from .callbacks import EpisodeLoggerCallback
from .config import EnvironmentConfig, SuccessCriteria, TrainingConfig
from .environment import DPDirectRLEnv


def make_environment(
    training_config: TrainingConfig,
    rank: int,
    training: bool,
    success: SuccessCriteria | None = None,
    seed_offset: int = 0,
):
    def initialize():
        environment_config = EnvironmentConfig(
            time_step=0.1,
            episode_steps=(
                training_config.episode_steps
                if training
                else training_config.evaluation_episode_steps
            ),
            training=training,
            measurement_noise=True,
            disturbance=training_config.disturbance,
            success=SuccessCriteria() if success is None else success,
        )
        environment = DPDirectRLEnv(environment_config)
        environment.reset(seed=training_config.seed + seed_offset + rank)
        return environment

    return initialize


def make_vector_environments(
    config: TrainingConfig,
    success: SuccessCriteria | None = None,
    seed_offset: int = 0,
):
    training_environment = VecMonitor(
        DummyVecEnv(
            [
                make_environment(config, rank, True, success, seed_offset)
                for rank in range(config.number_of_environments)
            ]
        )
    )
    evaluation_environment = VecMonitor(
        DummyVecEnv(
            [
                make_environment(
                    config,
                    0,
                    False,
                    success,
                    seed_offset + 1000,
                )
            ]
        )
    )
    return training_environment, evaluation_environment


def build_model(config: TrainingConfig, environment) -> TQC:
    policy_arguments = {
        "features_extractor_class": DemandActuatorAttentionExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [256, 256, 256],
    }
    return TQC(
        policy="MlpPolicy",
        env=environment,
        learning_rate=config.learning_rate,
        buffer_size=config.replay_buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_frequency,
        gradient_steps=config.gradient_steps,
        ent_coef=config.entropy_coefficient,
        top_quantiles_to_drop_per_net=config.top_quantiles_to_drop_per_net,
        policy_kwargs=policy_arguments,
        verbose=0,
        seed=config.seed,
        device=config.device,
    )


def make_callbacks(
    config: TrainingConfig,
    evaluation_environment,
    log_filename: str,
    best_directory: str,
    evaluation_directory: str,
    checkpoint_directory: str,
    checkpoint_prefix: str,
):
    output = Path(config.output_dir)
    return [
        EvalCallback(
            evaluation_environment,
            best_model_save_path=str(output / best_directory),
            log_path=str(output / evaluation_directory),
            eval_freq=config.evaluation_frequency,
            n_eval_episodes=config.evaluation_episodes,
            deterministic=True,
            render=False,
            verbose=0,
        ),
        CheckpointCallback(
            save_freq=max(
                config.checkpoint_frequency // config.number_of_environments,
                1,
            ),
            save_path=str(output / checkpoint_directory),
            name_prefix=checkpoint_prefix,
        ),
        EpisodeLoggerCallback(
            print_every_episodes=10,
            csv_path=str(output / log_filename),
            rolling_window=100,
        ),
    ]


def find_pretrained_model(output_dir: str, explicit_path: str | None = None) -> str:
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Model does not exist: {explicit_path}")
        return explicit_path
    candidates = [
        Path(output_dir) / "DAATQC_final.zip",
        Path(output_dir) / "best_model" / "best_model.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "No pretrained DAATQC model was found. Run train.py first or pass --model-path."
    )
