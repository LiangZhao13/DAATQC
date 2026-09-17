"""Heterogeneous six-thruster propulsion model."""

from __future__ import annotations

import numpy as np


class ThrusterSystem:
    def __init__(self) -> None:
        self.vessel_length = 76.2
        self.time_constant = 1.0
        self.maximum_speed = np.array([250, 250, 250, 250, 160, 160], dtype=np.float64)
        self.thrust_coefficients = np.diag([3.2, 3.2, 3.2, 3.2, 31.2, 31.2])
        self.types = ("T", "T", "T", "T", "M", "M")
        self.longitudinal_positions = (30.0, 22.0, -22.0, -30.0, -38.1, -38.1)
        self.transverse_positions = (0.0, 0.0, 0.0, 0.0, 8.0, -8.0)
        self.configuration_matrix = self._build_configuration_matrix()

    def _build_configuration_matrix(self) -> np.ndarray:
        matrix = np.zeros((3, len(self.types)), dtype=np.float64)
        for index, thruster_type in enumerate(self.types):
            x = self.longitudinal_positions[index]
            y = self.transverse_positions[index]
            if thruster_type == "M":
                matrix[:, index] = [1.0, 0.0, -y]
            elif thruster_type == "T":
                matrix[:, index] = [0.0, 1.0, x]
            else:
                azimuth = float(thruster_type)
                matrix[:, index] = [
                    np.cos(azimuth),
                    np.sin(azimuth),
                    x * np.sin(azimuth) - y * np.cos(azimuth),
                ]
        return matrix

    def saturate(self, rotational_speed: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(rotational_speed, dtype=np.float64),
            -self.maximum_speed,
            self.maximum_speed,
        )

    def get_force(self, rotational_speed: np.ndarray) -> np.ndarray:
        rotational_speed = np.asarray(rotational_speed, dtype=np.float64)
        signed_square = (np.abs(rotational_speed) * rotational_speed).reshape(-1, 1)
        return (self.configuration_matrix @ self.thrust_coefficients @ signed_square).ravel()

