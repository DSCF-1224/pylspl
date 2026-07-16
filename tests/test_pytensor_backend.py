"""Tests for the PyTensor backend."""

import numpy as np
import pytensor
import pytensor.tensor.type as ptt
import pytest

import utils

from pylspl.pytensor_backend import fit as fit_lspl
from pylspl.result import Vector3D


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("num_points", range(3, 11))
def test_fit_exact_plane(axis: str, num_points: int) -> None:
    """
    For points on the {axis}=0 plane,
    the normal should point along {axis} and the flatness should be 0.
    """

    coords = utils.make_axis_aligned_coords(axis=axis, num_points=num_points)

    result = fit_lspl(x=coords["x"], y=coords["y"], z=coords["z"])

    normal_x, normal_y, normal_z, flatness = \
        pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
            [],
            [result.normal.x, result.normal.y, result.normal.z, result.flatness]
        )()

    normal = {"x": normal_x, "y": normal_y, "z": normal_z}

    # The normal should be parallel to the {axis} axis (sign is undefined)
    utils.assert_normal_aligned_with_axis(normal, axis)

    # The flatness should be zero
    assert flatness == pytest.approx(0.0)


@pytest.mark.parametrize("num_points", range(3, 11))
@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_tilted_plane(num_points: int, seed: int) -> None:
    """
    A randomly oriented plane should be fitted correctly.
    """

    x, y, z, desired_normal = utils.make_tilted_plane_coords(
        num_points=num_points,
        seed=seed
    )

    result = fit_lspl(x=x, y=y, z=z)

    normal_x, normal_y, normal_z, flatness = \
        pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
            [],
            [result.normal.x, result.normal.y, result.normal.z, result.flatness]
        )()

    normal = Vector3D(x=normal_x, y=normal_y, z=normal_z)

    assert abs(desired_normal.dot(normal)) == pytest.approx(1.0)

    # The flatness should be zero
    assert flatness == pytest.approx(0.0)


@pytest.mark.parametrize("num_base_points", range(3, 11))
@pytest.mark.parametrize("seed", range(10))
def test_flatness_matches_known_value(seed: int, num_base_points: int) -> None:
    """
    For a point set whose exact flatness is known by construction,
    the fitted flatness should match.
    """

    rng = np.random.default_rng(seed=seed)

    x, y, z, delta = utils.make_mirrored_points(rng, num_base_points)

    assert fit_lspl(x=x, y=y, z=z).flatness.eval() == pytest.approx(2 * delta)


@pytest.mark.parametrize(
    "x_len, y_len, z_len",
    [
        (3, 4, 4), (4, 3, 4), (4, 4, 3),
        (5, 4, 4), (4, 5, 4), (4, 4, 5),
        (3, 4, 5), (4, 5, 3), (5, 3, 4),
        (5, 4, 3), (4, 3, 5), (3, 5, 4)
    ],
)
def test_mismatched_dynamic_length(x_len: int, y_len: int, z_len: int) -> None:
    """
    When lengths are not statically known, a mismatch should not raise
    at fit() call time, but should raise when the graph is evaluated.
    """

    x = ptt.vector("x")
    y = ptt.vector("y")
    z = ptt.vector("z")

    # should not raise here
    result = fit_lspl(x, y, z)

    fn = pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
        [x, y, z],
        [result.point.x, result.normal.x, result.flatness]
    )

    with pytest.raises(AssertionError, match="must have the same length"):
        fn(np.zeros(x_len), np.zeros(y_len), np.zeros(z_len))


@pytest.mark.parametrize(
    "x_len, y_len, z_len",
    [
        (3, 4, 4), (4, 3, 4), (4, 4, 3),
        (5, 4, 4), (4, 5, 4), (4, 4, 5),
        (3, 4, 5), (4, 5, 3), (5, 3, 4),
        (5, 4, 3), (4, 3, 5), (3, 5, 4)
    ],
)
def test_mismatched_static_length(x_len: int, y_len: int, z_len: int) -> None:
    """
    Reject points with mismatched coordinate lengths.
    """
    with pytest.raises(ValueError, match="must have the same length"):
        fit_lspl(x=np.zeros(x_len), y=np.zeros(y_len), z=np.zeros(z_len))


# pylint: disable=duplicate-code
@pytest.mark.parametrize("num_points", range(0, 3))
def test_requires_at_least_three_points(num_points: int) -> None:
    """Reject fewer than three points."""
    with pytest.raises(ValueError, match="at least 3 points are required"):
        fit_lspl(
            x=np.zeros(num_points),
            y=np.zeros(num_points),
            z=np.zeros(num_points)
        )
