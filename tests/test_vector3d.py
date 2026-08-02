"""Tests for pylspl.result.Vector3D."""

import numpy as np
import pytest

from pylspl.result import Vector3D


rng = np.random.default_rng(seed=42)


def test_vector3d_mul() -> None:
    """__mul__ should scale each component by the scalar."""

    for _ in range(5):

        coord = rng.uniform(size=3)
        scalar = rng.uniform(low=-5.0, high=5.0)

        vector = Vector3D(x=coord[0], y=coord[1], z=coord[2])
        scaled_vector = vector * scalar

        assert scaled_vector.x == pytest.approx(vector.x * scalar)
        assert scaled_vector.y == pytest.approx(vector.y * scalar)
        assert scaled_vector.z == pytest.approx(vector.z * scalar)


def test_vector3d_sub() -> None:
    """__sub__ should return the component-wise difference."""

    for _ in range(5):

        a_coord = rng.uniform(size=3)
        b_coord = rng.uniform(size=3)

        a = Vector3D(x=a_coord[0], y=a_coord[1], z=a_coord[2])
        b = Vector3D(x=b_coord[0], y=b_coord[1], z=b_coord[2])

        diff = a - b

        assert diff.x == pytest.approx(a.x - b.x)
        assert diff.y == pytest.approx(a.y - b.y)
        assert diff.z == pytest.approx(a.z - b.z)


def test_vector3d_normalize_zero_vector_raises() -> None:
    """normalize() should raise ZeroDivisionError for a zero vector."""

    with pytest.raises(ZeroDivisionError):
        Vector3D(x=0.0, y=0.0, z=0.0).normalize()
