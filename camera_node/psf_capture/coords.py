"""The single place where z (source-to-diffuser distance) meets slider coordinates.

  z      : distance from the diffuser plane to the point source, mm (10 at home, 340 max)
  steps  : motor steps away from home (>= 0); this is what the backend tracks
  slider : the Arduino's position in mm. It zeros at the front switch and its
           MOVE/POSITION sign is opposite to z: moving away from the camera is negative.
"""

from .config import STEPS_PER_MM, Z_HOME_MM, Z_MAX_MM


def z_to_steps(z_mm: float) -> int:
    """Nearest whole motor step to the requested z."""
    return round((z_mm - Z_HOME_MM) * STEPS_PER_MM)


def steps_to_z(steps: int) -> float:
    return round(Z_HOME_MM + steps / STEPS_PER_MM, 6)


def round_z(z_mm: float) -> float:
    """z as actually reachable by the motor."""
    return round(steps_to_z(z_to_steps(z_mm)), 6)


def steps_to_slider_mm(steps: float) -> float:
    return -steps / STEPS_PER_MM


def slider_mm_to_steps(slider_mm: float) -> int:
    return round(-slider_mm * STEPS_PER_MM)


def in_range(z_mm: float) -> bool:
    return Z_HOME_MM - 1e-9 <= z_mm <= Z_MAX_MM + 1e-9
