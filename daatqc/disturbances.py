"""Ocean-current, stochastic-wind, and low-frequency wave-drift models."""

from __future__ import annotations

import numpy as np

from .config import DisturbanceConfig
from .utils import smallest_signed_angle


def current_velocity_body(
    speed: float,
    direction_rad: float,
    heading_rad: float,
) -> np.ndarray:
    return np.array(
        [
            speed * np.cos(direction_rad - heading_rad),
            speed * np.sin(direction_rad - heading_rad),
            0.0,
        ],
        dtype=np.float64,
    )


class EnvironmentalDisturbance:
    """Wind and JONSWAP-based low-frequency wave forces.

    The model follows the reference ``TQC_energy_env`` notebook. Its output is
    added to the thruster generalized force, but wind and wave states are not
    appended to the policy observation.
    """

    def __init__(self, config: DisturbanceConfig, vessel_length: float = 76.2) -> None:
        self.config = config
        self.length = vessel_length
        self.gravity = 9.81
        self.air_density = 1.225
        self.frontal_area = 350.0
        self.lateral_area = 900.0
        self.wind_pressure_center_x = 0.05 * self.length
        self.wave_pressure_center_x = 0.15 * self.length
        self.wave_components = 32
        self.omega_min = 0.25
        self.omega_max = 2.50
        self.jonswap_gamma = 3.3
        self.force_clip = np.array([1.0e5, 1.0e5, 3.0e6], dtype=np.float64)
        self.wind_speed_mean = 0.0
        self.wind_speed = 0.0
        self.wind_direction = 0.0
        self.wave_height = 0.0
        self.wave_period = 6.0
        self.wave_direction = 0.0
        self.wave_low_frequency_state = 0.0
        self.omega = np.empty(0)
        self.wave_amplitude = np.empty(0)
        self.wave_phase = np.empty(0)

    def reset(self, rng: np.random.Generator, options: dict | None = None) -> None:
        options = {} if options is None else options
        self.wind_speed_mean = float(options.get("wind_speed_mean", self.config.wind_speed_mean))
        self.wind_speed = self.wind_speed_mean
        self.wind_direction = float(
            options.get("wind_direction", np.deg2rad(self.config.wind_direction_deg))
        )
        self.wave_height = float(
            options.get("significant_wave_height", self.config.significant_wave_height)
        )
        self.wave_period = float(options.get("wave_peak_period", self.config.wave_peak_period))
        self.wave_direction = float(
            options.get("wave_direction", np.deg2rad(self.config.wave_direction_deg))
        )
        self.wave_low_frequency_state = 0.0
        self._initialize_wave_spectrum(rng)

    def _jonswap_spectrum(self, omega: np.ndarray) -> np.ndarray:
        peak = 2.0 * np.pi / max(self.wave_period, 1e-6)
        sigma = np.where(omega <= peak, 0.07, 0.09)
        concentration = np.exp(-((omega / peak - 1.0) ** 2) / (2.0 * sigma**2))
        raw = (
            self.gravity**2
            * omega**-5.0
            * np.exp(-1.25 * (peak / omega) ** 4)
            * self.jonswap_gamma**concentration
        )
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        raw_moment = integrate(raw, omega)
        target_moment = self.wave_height**2 / 16.0
        if raw_moment <= 1e-12 or target_moment <= 1e-12:
            return np.zeros_like(omega)
        return raw * target_moment / raw_moment

    def _initialize_wave_spectrum(self, rng: np.random.Generator) -> None:
        self.omega = np.linspace(self.omega_min, self.omega_max, self.wave_components)
        frequency_step = self.omega[1] - self.omega[0]
        spectrum = self._jonswap_spectrum(self.omega)
        self.wave_amplitude = np.sqrt(2.0 * spectrum * frequency_step)
        self.wave_phase = rng.uniform(0.0, 2.0 * np.pi, size=self.wave_components)

    @staticmethod
    def _rotation_2d(heading: float) -> np.ndarray:
        return np.array(
            [[np.cos(heading), -np.sin(heading)],
             [np.sin(heading), np.cos(heading)]],
            dtype=np.float64,
        )

    def _relative_wind_body(self, eta: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        rotation = self._rotation_2d(float(eta[2]))
        wind_inertial = self.wind_speed * np.array(
            [np.cos(self.wind_direction), np.sin(self.wind_direction)]
        )
        vessel_inertial = rotation @ np.asarray(velocity[:2], dtype=np.float64)
        return rotation.T @ (wind_inertial - vessel_inertial)

    def _update_wind(self, rng: np.random.Generator, time_step: float) -> None:
        if self.wind_speed_mean <= 1e-9:
            self.wind_speed = 0.0
            return
        noise = self.config.wind_sigma * np.sqrt(time_step) * rng.standard_normal()
        restoring = (
            time_step / self.config.wind_time_constant
            * (self.wind_speed_mean - self.wind_speed)
        )
        self.wind_speed = float(
            np.clip(
                self.wind_speed + restoring + noise,
                0.0,
                self.config.wind_speed_max,
            )
        )

    def _wind_force(self, eta: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        if self.wind_speed <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        relative_wind = self._relative_wind_body(eta, velocity)
        longitudinal, transverse = relative_wind
        relative_speed = np.hypot(longitudinal, transverse)
        if relative_speed <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        angle = np.arctan2(transverse, longitudinal)
        cosine, sine = np.cos(angle), np.sin(angle)
        coefficient_x = 0.65 + 0.35 * abs(cosine)
        coefficient_y = 0.80 + 0.45 * abs(sine)
        dynamic_pressure = 0.5 * self.air_density * relative_speed**2
        force_x = (
            self.config.wind_scale * dynamic_pressure * self.frontal_area
            * coefficient_x * cosine * abs(cosine)
        )
        force_y = (
            self.config.wind_scale * dynamic_pressure * self.lateral_area
            * coefficient_y * sine * abs(sine)
        )
        moment = self.wind_pressure_center_x * force_y
        return np.array([force_x, force_y, moment], dtype=np.float64)

    def _wave_force(self, eta: np.ndarray, time: float, time_step: float) -> np.ndarray:
        if self.wave_height <= 1e-9:
            return np.zeros(3, dtype=np.float64)
        elevation = float(
            np.sum(self.wave_amplitude * np.cos(self.omega * time + self.wave_phase))
        )
        variance = max(self.wave_height**2 / 16.0, 1e-8)
        raw_energy = elevation**2 / variance - 1.0
        alpha = time_step / (self.config.wave_low_pass_time_constant + time_step)
        self.wave_low_frequency_state = float(
            np.clip(
                (1.0 - alpha) * self.wave_low_frequency_state + alpha * raw_energy,
                -0.8,
                1.5,
            )
        )
        mean_force = self.config.wave_drift_coefficient * self.wave_height**2
        force = self.config.wave_scale * mean_force * (
            1.0 + self.config.wave_low_frequency_gain * self.wave_low_frequency_state
        )
        relative_direction = smallest_signed_angle(self.wave_direction - eta[2])
        cosine, sine = np.cos(relative_direction), np.sin(relative_direction)
        force_x = force * cosine * abs(cosine)
        force_y = force * sine * abs(sine)
        return np.array(
            [force_x, force_y, self.wave_pressure_center_x * force_y],
            dtype=np.float64,
        )

    def step(
        self,
        eta: np.ndarray,
        velocity: np.ndarray,
        time: float,
        time_step: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict]:
        if not self.config.enable_wind_wave:
            zero = np.zeros(3, dtype=np.float64)
            return zero, self._info(zero, zero, zero)
        self._update_wind(rng, time_step)
        wind_force = self._wind_force(eta, velocity)
        wave_force = self._wave_force(eta, time, time_step)
        total = np.clip(wind_force + wave_force, -self.force_clip, self.force_clip)
        return total, self._info(wind_force, wave_force, total)

    def _info(
        self,
        wind_force: np.ndarray,
        wave_force: np.ndarray,
        total: np.ndarray,
    ) -> dict:
        return {
            "tau_wind": wind_force.copy(),
            "tau_wave": wave_force.copy(),
            "tau_env": total.copy(),
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "significant_wave_height": self.wave_height,
            "wave_peak_period": self.wave_period,
            "wave_direction": self.wave_direction,
            "wave_low_frequency_state": self.wave_low_frequency_state,
        }
