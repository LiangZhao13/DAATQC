"""Three-degree-of-freedom low-speed supply-vessel model."""

from __future__ import annotations

import numpy as np


class SupplyVessel:
    def __init__(self) -> None:
        self.length = 76.2
        self.gravity = 9.81
        self.mass = 6000e3
        self._initialize_matrices()

    def _initialize_matrices(self) -> None:
        transform = np.diag([1.0, 1.0, 1.0 / self.length])
        inverse_transform = np.diag([1.0, 1.0, self.length])
        mass_bis = np.array(
            [[1.1274, 0.0, 0.0], [0.0, 1.8902, -0.0744],
             [0.0, -0.0744, 0.1278]],
            dtype=np.float64,
        )
        damping_bis = np.array(
            [[0.0358, 0.0, 0.0], [0.0, 0.1183, -0.0124],
             [0.0, -0.0041, 0.0308]],
            dtype=np.float64,
        )
        scale = inverse_transform @ inverse_transform
        self.mass_matrix = self.mass * scale @ (
            transform @ mass_bis @ inverse_transform
        )
        self.damping_matrix = self.mass * scale @ (
            np.sqrt(self.gravity / self.length)
            * transform @ damping_bis @ inverse_transform
        )
        self.inverse_mass_matrix = np.linalg.inv(self.mass_matrix)

    def get_derivatives(
        self,
        eta: np.ndarray,
        relative_velocity: np.ndarray,
        generalized_force: np.ndarray,
    ) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64).reshape(3, 1)
        relative_velocity = np.asarray(relative_velocity, dtype=np.float64).reshape(3, 1)
        generalized_force = np.asarray(generalized_force, dtype=np.float64).reshape(3, 1)
        psi = eta[2, 0]
        yaw_rotation = np.array(
            [[np.cos(psi), -np.sin(psi), 0.0],
             [np.sin(psi), np.cos(psi), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        system_matrix = np.vstack(
            [
                np.hstack([np.zeros((3, 3)), yaw_rotation]),
                np.hstack([
                    np.zeros((3, 3)),
                    -self.inverse_mass_matrix @ self.damping_matrix,
                ]),
            ]
        )
        input_matrix = np.vstack([np.zeros((3, 3)), self.inverse_mass_matrix])
        state = np.vstack([eta, relative_velocity])
        return (system_matrix @ state + input_matrix @ generalized_force).ravel()

