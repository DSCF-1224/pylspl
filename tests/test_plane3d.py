"""Tests for pylspl.result.Plane3D."""

import itertools

import pytest

import utils

from pylspl.result import Vector3D


VECTOR3D_CASES = list(itertools.product([-2.0, 0.0, 2.0], repeat=3))


@pytest.mark.parametrize("x, y, z", VECTOR3D_CASES)
def test_plane3d_signed_distance(x: float, y: float, z: float) -> None:
    """
    signed_distance should return the perpendicular distance from a
    point to the plane, positive on the side the normal points to.
    """

    vector = Vector3D(x=x, y=y, z=z)

    assert utils.PLANE_Z0.signed_distance(vector) == pytest.approx(z)
    assert utils.PLANE_ZP1.signed_distance(vector) == pytest.approx(z - 1.0)
    assert utils.PLANE_ZN1.signed_distance(vector) == pytest.approx(z + 1.0)
