"""Tests for pylspl.result."""

import itertools

import pytest

from pylspl.result import Plane3D, Vector3D


VECTOR3D_CASES = list(itertools.product([-2.0, 0.0, 2.0], repeat=3))

NORMAL_VECTOR = Vector3D(x=0.0, y=0.0, z=1.0)

PLANE_Z0 = Plane3D(point=Vector3D(x=0.0, y=0.0, z=0.0), normal=NORMAL_VECTOR)
PLANE_ZP1 = Plane3D(point=Vector3D(x=0.0, y=0.0, z=1.0), normal=NORMAL_VECTOR)
PLANE_ZN1 = Plane3D(point=Vector3D(x=0.0, y=0.0, z=-1.0), normal=NORMAL_VECTOR)


@pytest.mark.parametrize("x, y, z", VECTOR3D_CASES)
def test_plane3d_signed_distance(x: float, y: float, z: float) -> None:
    """
    signed_distance should return the perpendicular distance from a
    point to the plane, positive on the side the normal points to.
    """

    vector = Vector3D(x=x, y=y, z=z)

    assert PLANE_Z0.signed_distance(vector) == pytest.approx(z)
    assert PLANE_ZP1.signed_distance(vector) == pytest.approx(z - 1.0)
    assert PLANE_ZN1.signed_distance(vector) == pytest.approx(z + 1.0)


def test_vector3d_normalize_zero_vector_raises() -> None:
    """normalize() should raise ZeroDivisionError for a zero vector."""
    with pytest.raises(ZeroDivisionError):
        Vector3D(x=0.0, y=0.0, z=0.0).normalize()
