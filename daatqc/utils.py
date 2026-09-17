"""Small numerical helpers shared by the simulation modules."""

from __future__ import annotations

import numpy as np


def smallest_signed_angle(angle: float | np.ndarray, unit: str = "rad"):
    if unit == "rad":
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
    if unit == "deg":
        return (angle + 180.0) % 360.0 - 180.0
    raise ValueError("unit must be 'rad' or 'deg'")


def rotation_zyx(phi: float, theta: float, psi: float) -> np.ndarray:
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cpsi * ctheta, -spsi * cphi + cpsi * stheta * sphi,
             spsi * sphi + cpsi * cphi * stheta],
            [spsi * ctheta, cpsi * cphi + sphi * stheta * spsi,
             -cpsi * sphi + stheta * spsi * cphi],
            [-stheta, ctheta * sphi, ctheta * cphi],
        ],
        dtype=np.float64,
    )


# Short aliases retained for readers familiar with the source notebook.
ssa = smallest_signed_angle
Rzyx = rotation_zyx

