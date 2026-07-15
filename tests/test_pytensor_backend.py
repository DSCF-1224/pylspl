"""Tests for the PyTensor backend."""

import pytensor.compile.function as ptf
import pytest

import utils

from pylspl.pytensor_backend import fit as fit_lspl


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("num_points", range(3, 11))
def test_fit_exact_plane(axis: str, num_points: int) -> None:
    """
    For points on the {axis}=0 plane,
    the normal should point along {axis} and the flatness should be 0.
    """

    coords = utils.make_axis_aligned_coords(axis=axis, num_points=num_points)

    result = fit_lspl(x=coords["x"], y=coords["y"], z=coords["z"])

    normal_x, normal_y, normal_z, flatness = ptf.function(
        [],
        [result.normal.x, result.normal.y, result.normal.z, result.flatness]
    )()

    normal = {"x": normal_x, "y": normal_y, "z": normal_z}

    # The normal should be parallel to the {axis} axis (sign is undefined)
    utils.assert_normal_aligned_with_axis(normal, axis)

    # The flatness should be zero
    assert flatness == pytest.approx(0.0)
