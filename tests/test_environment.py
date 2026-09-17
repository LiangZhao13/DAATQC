"""Fast checks for state dimensions and disturbance integration."""

import numpy as np

from daatqc.config import DisturbanceConfig, EnvironmentConfig
from daatqc.environment import DPDirectRLEnv


def test_default_environment_keeps_14_dimensional_observation():
    environment = DPDirectRLEnv(
        EnvironmentConfig(training=False, measurement_noise=False, episode_steps=5)
    )
    observation, _ = environment.reset(seed=42)
    assert observation.shape == (14,)
    assert np.allclose(observation[12:14], 0.0)
    next_observation, _, _, _, info = environment.step(np.zeros(6))
    assert next_observation.shape == (14,)
    assert np.allclose(info["tau_env"], 0.0)
    environment.close()


def test_current_wind_and_wave_act_only_through_dynamics():
    disturbance = DisturbanceConfig(
        current_speed=0.5,
        wind_speed_mean=3.0,
        significant_wave_height=0.3,
    )
    environment = DPDirectRLEnv(
        EnvironmentConfig(
            training=False,
            measurement_noise=False,
            episode_steps=5,
            disturbance=disturbance,
        )
    )
    observation, _ = environment.reset(seed=42)
    assert observation.shape == (14,)
    assert np.allclose(observation[12:14], 0.0)
    next_observation, _, _, _, info = environment.step(np.zeros(6))
    assert np.allclose(next_observation[12:14], 0.0)
    assert not np.allclose(info["current_body"], 0.0)
    assert info["tau_env"].shape == (3,)
    assert not np.allclose(info["tau_env"], 0.0)
    environment.close()


def test_legacy_current_channels_can_be_enabled_explicitly():
    environment = DPDirectRLEnv(
        EnvironmentConfig(
            training=False,
            measurement_noise=False,
            episode_steps=5,
            expose_current_to_policy=True,
            disturbance=DisturbanceConfig(current_speed=0.5),
        )
    )
    observation, _ = environment.reset(seed=42)
    assert observation.shape == (14,)
    assert not np.allclose(observation[12:14], 0.0)
    environment.close()


def test_hidden_current_still_changes_vessel_dynamics():
    base_config = dict(
        training=False,
        measurement_noise=False,
        episode_steps=5,
    )
    calm = DPDirectRLEnv(
        EnvironmentConfig(**base_config, disturbance=DisturbanceConfig())
    )
    current = DPDirectRLEnv(
        EnvironmentConfig(
            **base_config,
            disturbance=DisturbanceConfig(current_speed=0.5),
        )
    )
    reset_options = {
        "eta_ref": np.zeros(3),
        "eta0": np.zeros(3),
        "nu0": np.zeros(3),
        "n0": np.zeros(6),
    }
    calm.reset(seed=42, options=reset_options)
    current.reset(seed=42, options=reset_options)
    calm_observation, *_ = calm.step(np.zeros(6))
    current_observation, *_ = current.step(np.zeros(6))
    assert np.allclose(calm_observation[12:14], 0.0)
    assert np.allclose(current_observation[12:14], 0.0)
    assert not np.allclose(calm.velocity, current.velocity)
    calm.close()
    current.close()
