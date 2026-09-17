"""Configuration objects for DAATQC training and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DisturbanceConfig:
    """Environmental conditions applied only through the vessel dynamics.

    The defaults intentionally disable current, wind, and waves for the
    reported training setup. Non-zero values may be supplied for robustness
    experiments without changing the 14-dimensional observation vector.
    """

    current_speed: float = 0.0
    current_direction_deg: float = 0.0
    wind_speed_mean: float = 0.0
    wind_direction_deg: float = 0.0
    significant_wave_height: float = 0.0
    wave_peak_period: float = 6.0
    wave_direction_deg: float = 0.0
    enable_wind_wave: bool = True

    wind_scale: float = 0.10
    wind_time_constant: float = 40.0
    wind_sigma: float = 0.08
    wind_speed_max: float = 8.0
    wave_scale: float = 0.10
    wave_drift_coefficient: float = 4.0e4
    wave_low_frequency_gain: float = 0.35
    wave_low_pass_time_constant: float = 25.0


@dataclass(frozen=True)
class SuccessCriteria:
    position_threshold: float = 0.8
    heading_threshold_deg: float = 5.0
    speed_threshold: float = 0.10
    hold_steps: int = 30


@dataclass(frozen=True)
class EnvironmentConfig:
    time_step: float = 0.1
    episode_steps: int = 1200
    training: bool = True
    measurement_noise: bool = True
    expose_current_to_policy: bool = False
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    success: SuccessCriteria = field(default_factory=SuccessCriteria)


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str = "./drl_dp_DAATQC_results"
    seed: int = 42
    number_of_environments: int = 4
    total_timesteps: int = 1_200_000
    episode_steps: int = 1200
    evaluation_episode_steps: int = 1500
    learning_rate: float = 3e-4
    replay_buffer_size: int = 300_000
    learning_starts: int = 10_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.995
    train_frequency: int = 1
    gradient_steps: int = 1
    entropy_coefficient: float = 0.05
    top_quantiles_to_drop_per_net: int = 2
    evaluation_frequency: int = 10_000
    evaluation_episodes: int = 5
    checkpoint_frequency: int = 50_000
    device: str = "auto"
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
