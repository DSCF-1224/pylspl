"""Result models for plane fitting algorithms."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Vector3D:
    """A 3D Vector"""

    x: Any
    y: Any
    z: Any

    def dot(self, other: "Vector3D") -> Any:
        """Return the dot product with another vector."""
        return (self.x * other.x) + \
               (self.y * other.y) + \
               (self.z * other.z)

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        """Return the difference with another vector."""
        return Vector3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
