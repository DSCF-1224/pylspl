"""Result models for plane fitting algorithms."""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Vector3D:
    """A 3D Vector"""

    x: Any
    y: Any
    z: Any

    def __mul__(self, other: Any) -> "Vector3D":
        """Return this vector scaled by a scalar."""
        return Vector3D(
            self.x * other,
            self.y * other,
            self.z * other
        )

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        """Return the difference with another vector."""
        return Vector3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __truediv__(self, divisor: float) -> "Vector3D":
        """
        Return this vector scaled by the reciprocal of divisor.

        Raises
        ------
        ZeroDivisionError
            If divisor is zero.
        """
        return self * (1 / divisor)

    def dot(self, other: "Vector3D") -> Any:
        """Return the dot product with another vector."""
        return (self.x * other.x) + \
               (self.y * other.y) + \
               (self.z * other.z)

    def norm(self) -> Any:
        """Return the Euclidean norm (magnitude) of this vector."""
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vector3D":
        """
        Return a unit vector in the same direction as this vector.

        Raises
        ------
        ZeroDivisionError
            If this vector has zero norm.
        """
        return self / self.norm()


@dataclass(frozen=True)
class Plane3D:
    """A plane in 3D space, defined by a point on the plane and a normal vector."""

    point: Vector3D
    normal: Vector3D

    def signed_distance(self, point: Vector3D) -> Any:
        """Return the signed distance from designated point to this plane."""
        return (point - self.point).dot(self.normal)


@dataclass(frozen=True)
class FittedPlane3D(Plane3D):
    """Plane obtained from a fitting algorithm, with its evaluated flatness."""

    flatness: Any
