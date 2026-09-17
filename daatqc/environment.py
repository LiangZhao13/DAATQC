"""Gymnasium environment for end-to-end dynamic positioning."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import EnvironmentConfig
from .disturbances import EnvironmentalDisturbance, current_velocity_body
from .thrusters import ThrusterSystem
from .utils import rotation_zyx, smallest_signed_angle
from .vessel import SupplyVessel


class DPDirectRLEnv(gym.Env):
    """Directly map a 14-D vessel observation to six thruster commands.

    Observation order (unchanged from the DAATQC notebook): body-frame pose
    error (3), vessel velocity (3), normalized actual thruster speed (6), and
    two reserved current channels. By default, all environmental channels are
    zero so that current, wind, and waves affect decisions only indirectly via
    the vessel response. ``expose_current_to_policy=True`` restores the original
    notebook behaviour without changing the 14-D shape.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        super().__init__()
        self.config = EnvironmentConfig() if config is None else config
        self.time_step = self.config.time_step
        self.episode_steps = self.config.episode_steps
        self.training = self.config.training
        self.vessel = SupplyVessel()
        self.thrusters = ThrusterSystem()
        self.environmental_disturbance = EnvironmentalDisturbance(
            self.config.disturbance, self.vessel.length
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        observation_high = np.array(
            [
                1e3, 1e3, np.pi,
                20.0, 20.0, 5.0,
                1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                5.0, 5.0,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -observation_high, observation_high, dtype=np.float32
        )

        self.energy_weight = np.array([1, 1, 1, 1, 10, 10], dtype=np.float64)
        self.energy_penalty_coefficient = 2.0
        self.eta = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.thruster_speed = np.zeros(6, dtype=np.float64)
        self.previous_action = np.zeros(6, dtype=np.float64)
        self.reference = np.zeros(3, dtype=np.float64)
        self.current_speed = 0.0
        self.current_direction = 0.0
        self.step_count = 0
        self.success_steps = 0
        self.previous_distance = 0.0
        self.last_measurement = self.eta.copy()

    def _measurement(self) -> np.ndarray:
        measurement = self.eta.copy()
        if self.config.measurement_noise:
            measurement[0] += 0.01 * self.np_random.standard_normal()
            measurement[1] += 0.01 * self.np_random.standard_normal()
            measurement[2] += 0.0001 * self.np_random.standard_normal()
        measurement[2] = smallest_signed_angle(measurement[2])
        return measurement

    def _current_body(self) -> np.ndarray:
        return current_velocity_body(
            self.current_speed, self.current_direction, float(self.eta[2])
        )

    def _body_error(self, measurement: np.ndarray) -> np.ndarray:
        inertial_error = self.reference[:2] - measurement[:2]
        cosine, sine = np.cos(measurement[2]), np.sin(measurement[2])
        return np.array(
            [
                cosine * inertial_error[0] + sine * inertial_error[1],
                -sine * inertial_error[0] + cosine * inertial_error[1],
                smallest_signed_angle(self.reference[2] - measurement[2]),
            ],
            dtype=np.float64,
        )

    def _observation(self) -> np.ndarray:
        measurement = self._measurement()
        self.last_measurement = measurement.copy()
        current_observation = (
            self._current_body()[:2]
            if self.config.expose_current_to_policy
            else np.zeros(2, dtype=np.float64)
        )
        return np.concatenate(
            [
                self._body_error(measurement),
                self.velocity,
                self.thruster_speed / self.thrusters.maximum_speed,
                current_observation,
            ]
        ).astype(np.float32)

    def _sample_reference(self) -> np.ndarray:
        return np.array(
            [
                self.np_random.uniform(-5.0, 5.0),
                self.np_random.uniform(-5.0, 5.0),
                np.deg2rad(self.np_random.uniform(-90.0, 90.0)),
            ],
            dtype=np.float64,
        )

    def _sample_initial_state(self) -> np.ndarray:
        return np.array(
            [
                self.np_random.uniform(-8.0, 8.0),
                self.np_random.uniform(-8.0, 8.0),
                np.deg2rad(self.np_random.uniform(-60.0, 60.0)),
            ],
            dtype=np.float64,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = {} if options is None else options
        self.reference = np.array(
            options.get(
                "eta_ref",
                self._sample_reference()
                if self.training
                else np.array([0.0, 5.0, np.deg2rad(50.0)]),
            ),
            dtype=np.float64,
            copy=True,
        )
        self.eta = np.array(
            options.get(
                "eta0", self._sample_initial_state() if self.training else np.zeros(3)
            ),
            dtype=np.float64,
            copy=True,
        )
        self.velocity = np.array(
            options.get("nu0", np.zeros(3)), dtype=np.float64, copy=True
        )
        self.thruster_speed = np.array(
            options.get("n0", np.zeros(6)), dtype=np.float64, copy=True
        )
        self.current_speed = float(
            options.get("current_speed", options.get("Vc", self.config.disturbance.current_speed))
        )
        self.current_direction = float(
            options.get(
                "current_direction",
                options.get(
                    "betaVc", np.deg2rad(self.config.disturbance.current_direction_deg)
                ),
            )
        )
        self.environmental_disturbance.reset(self.np_random, options)
        self.previous_action = np.zeros(6, dtype=np.float64)
        self.step_count = 0
        self.success_steps = 0
        self.previous_distance = float(np.linalg.norm(self.reference[:2] - self.eta[:2]))
        observation = self._observation()
        info = {
            "eta_ref": self.reference.copy(),
            "current_speed": self.current_speed,
            "current_direction": self.current_direction,
            "wind_speed": self.environmental_disturbance.wind_speed,
            "significant_wave_height": self.environmental_disturbance.wave_height,
        }
        return observation, info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(6), -1.0, 1.0)
        commanded_speed = action * self.thrusters.maximum_speed
        self.thruster_speed = self.thrusters.saturate(self.thruster_speed)

        current_body = self._current_body()
        relative_velocity = self.velocity - current_body
        thruster_force = self.thrusters.get_force(self.thruster_speed)
        environment_force, disturbance_info = self.environmental_disturbance.step(
            self.eta,
            self.velocity,
            self.step_count * self.time_step,
            self.time_step,
            self.np_random,
        )
        total_force = thruster_force + environment_force
        derivative = self.vessel.get_derivatives(self.eta, relative_velocity, total_force)
        thruster_derivative = (
            commanded_speed - self.thruster_speed
        ) / self.thrusters.time_constant

        self.velocity += self.time_step * derivative[3:6]
        self.eta += self.time_step * (
            rotation_zyx(0.0, 0.0, float(self.eta[2])) @ self.velocity
        )
        self.eta[2] = smallest_signed_angle(self.eta[2])
        self.thruster_speed = self.thrusters.saturate(
            self.thruster_speed + self.time_step * thruster_derivative
        )
        self.step_count += 1

        observation = self._observation()
        body_error = observation[:3].astype(np.float64)
        normalized_speed = self.thruster_speed / self.thrusters.maximum_speed
        action_change = action - self.previous_action
        position_cost = abs(body_error[0]) + abs(body_error[1])
        heading_cost = abs(body_error[2])
        velocity_cost = (
            abs(self.velocity[0]) + abs(self.velocity[1]) + 0.5 * abs(self.velocity[2])
        )
        speed_cost = np.sum(np.abs(normalized_speed))
        smoothness_cost = np.sum(np.abs(action_change))
        distance = float(np.linalg.norm(self.reference[:2] - self.eta[:2]))
        path_reward = self.previous_distance - distance
        energy_cost = float(
            np.sum(self.energy_weight * np.abs(normalized_speed) ** 3)
            / np.sum(self.energy_weight)
        )
        energy_penalty = self.energy_penalty_coefficient * energy_cost
        reward = (
            -6.0 * position_cost
            -6.0 * heading_cost
            -0.2 * velocity_cost
            -0.02 * speed_cost
            -0.01 * smoothness_cost
            -0.4 * energy_penalty
            +5.0 * path_reward
        )

        position_error = float(np.linalg.norm(body_error[:2]))
        heading_error = float(abs(body_error[2]))
        speed_norm = float(np.linalg.norm(self.velocity))
        if position_error < 0.5 and heading_error < np.deg2rad(5.0):
            reward += 5.0
        if position_error < 0.2 and heading_error < np.deg2rad(2.0) and speed_norm < 0.05:
            reward += 10.0

        success = self.config.success
        if (
            position_error < success.position_threshold
            and heading_error < np.deg2rad(success.heading_threshold_deg)
            and speed_norm < success.speed_threshold
        ):
            self.success_steps += 1
        else:
            self.success_steps = 0

        terminated = False
        is_success = False
        if self.success_steps >= success.hold_steps:
            reward += 1000.0
            terminated = True
            is_success = True
        if (
            np.any(np.isnan(observation))
            or np.linalg.norm(self.eta[:2] - self.reference[:2]) > 120.0
            or speed_norm > 8.0
        ):
            reward -= 100.0
            terminated = True
        truncated = self.step_count >= self.episode_steps

        self.previous_action = action.copy()
        self.previous_distance = distance
        info = {
            "eta_ref": self.reference.copy(),
            "eta_meas": self.last_measurement.copy(),
            "n_c": commanded_speed.copy(),
            "n_prop": self.thruster_speed.copy(),
            "tau_thruster": thruster_force.copy(),
            "tau_env": environment_force.copy(),
            "tau_actual": total_force.copy(),
            "current_body": current_body.copy(),
            "pos_err": position_error,
            "yaw_err_deg": np.rad2deg(heading_error),
            "speed_norm": speed_norm,
            "success_steps": self.success_steps,
            "is_success": is_success,
            "global_dist": distance,
            "path_reward": path_reward,
            "energy_cost": energy_cost,
            "energy_penalty": energy_penalty,
            **disturbance_info,
        }
        return observation, float(reward), terminated, truncated, info
