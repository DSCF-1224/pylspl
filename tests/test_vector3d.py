"""Tests for pylspl.result.Vector3D."""

import numpy as np
import pytest

from pylspl.result import Vector3D


rng = np.random.default_rng(seed=42)


def _random_vector3d(low: float = 0.0, high: float = 1.0) -> Vector3D:
    """Return a Vector3D with components drawn from rng.uniform(low, high)."""

    coord = rng.uniform(low=low, high=high, size=3)
    return Vector3D(x=coord[0], y=coord[1], z=coord[2])


def test_vector3d_mul() -> None:
    """__mul__ should scale each component by the scalar."""

    for _ in range(5):

        scalar = rng.uniform(low=-5.0, high=5.0)
        vector = _random_vector3d()

        scaled_vector = vector * scalar

        assert scaled_vector.x == pytest.approx(vector.x * scalar)
        assert scaled_vector.y == pytest.approx(vector.y * scalar)
        assert scaled_vector.z == pytest.approx(vector.z * scalar)


def test_vector3d_sub() -> None:
    """__sub__ should return the component-wise difference."""

    for _ in range(5):

        a = _random_vector3d()
        b = _random_vector3d()

        diff = a - b

        assert diff.x == pytest.approx(a.x - b.x)
        assert diff.y == pytest.approx(a.y - b.y)
        assert diff.z == pytest.approx(a.z - b.z)


def test_vector3d_truediv() -> None:
    """__truediv__ should scale each component by the reciprocal of the divisor."""

    for _ in range(5):

        divisor = rng.uniform(low=0.1, high=5.0)
        vector = _random_vector3d()

        scaled_vector = vector / divisor

        assert scaled_vector.x == pytest.approx(vector.x / divisor)
        assert scaled_vector.y == pytest.approx(vector.y / divisor)
        assert scaled_vector.z == pytest.approx(vector.z / divisor)


def test_vector3d_truediv_by_zero_raises() -> None:
    """__truediv__ should raise ZeroDivisionError when divisor is zero."""

    for _ in range(5):
        with pytest.raises(ZeroDivisionError):
            _ = _random_vector3d() / 0.0


def test_vector3d_normalize_zero_vector_raises() -> None:
    """normalize() should raise ZeroDivisionError for a zero vector."""

    with pytest.raises(ZeroDivisionError):
        Vector3D(x=0.0, y=0.0, z=0.0).normalize()
