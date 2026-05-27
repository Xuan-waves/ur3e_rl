"""Compatibility imports for older scripts.

New code should import from ``real_teleop.impedance`` or
``real_teleop.impedance.config`` directly.
"""

from real_teleop.impedance.config import (  # noqa: F401
    DEFAULT_IMPEDANCE_TEST_CONFIG,
    ImpedanceProfile,
    ImpedanceRuntimeConfig,
    ImpedanceTestConfig,
    Vector3,
    Vector6,
)
