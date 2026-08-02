"""Tests for pylspl.result.Vector3D."""

import pytest

from pylspl.result import Vector3D


def test_vector3d_normalize_zero_vector_raises() -> None:
    """normalize() should raise ZeroDivisionError for a zero vector."""

    with pytest.raises(ZeroDivisionError):
        Vector3D(x=0.0, y=0.0, z=0.0).normalize()
