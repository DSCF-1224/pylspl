"""Tests for the PyTensor backend."""

import numpy as np
import pytensor
import pytensor.gradient as pg
import pytensor.tensor.basic as ptb
import pytensor.tensor.type as ptt
import pytensor.tensor.variable as ptv
import pytest

import utils

from pylspl.pytensor_backend import fit as fit_lspl
from pylspl.result import Vector3D
from pylspl._messages import MSG_MIN_POINTS, MSG_NOT_1D, MSG_SAME_LENGTH


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


@pytest.mark.parametrize("seed", range(5))
def test_flatness_gradient(seed: int) -> None:
    """
    The gradient of flatness with respect to x, y, z should match the
    numerical gradient (finite differences).
    """
    rng = np.random.default_rng(seed=seed)

    # A well-separated (non-degenerate) point set, so the eigenvalue
    # gradient stays away from the singular (repeated-eigenvalue) case.
    x0 = rng.uniform(low=-1.0, high=1.0, size=8)
    y0 = rng.uniform(low=-1.0, high=1.0, size=8)
    z0 = rng.normal(scale=0.05, size=8)

    def flatness_fn(x: ptv.TensorVariable, y: ptv.TensorVariable, z: ptv.TensorVariable):
        return fit_lspl(x, y, z).flatness

    pg.verify_grad(
        flatness_fn, [x0, y0, z0], rng=np.random.default_rng(seed=seed + 100)
    )


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


@pytest.mark.parametrize("x_len, y_len, z_len", utils.MISMATCHED_LENGTH_CASES)
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

    with pytest.raises(AssertionError, match=MSG_SAME_LENGTH):
        fn(np.zeros(x_len), np.zeros(y_len), np.zeros(z_len))


@pytest.mark.parametrize("x_len, y_len, z_len", utils.MISMATCHED_LENGTH_CASES)
def test_mismatched_static_length(x_len: int, y_len: int, z_len: int) -> None:
    """
    Reject points with mismatched coordinate lengths.
    """
    with pytest.raises(ValueError, match=MSG_SAME_LENGTH):
        fit_lspl(x=np.zeros(x_len), y=np.zeros(y_len), z=np.zeros(z_len))


@pytest.mark.parametrize("seed", range(5))
def test_normal_gradient(seed: int) -> None:
    """
    The gradient of normal with respect to x, y, z should match the
    numerically estimated gradient (finite differences).
    """
    rng = np.random.default_rng(seed=seed)

    # A well-separated (non-degenerate) point set, so the eigenvalue
    # gradient stays away from the singular (repeated-eigenvalue) case.
    x0 = rng.uniform(low=-1.0, high=1.0, size=8)
    y0 = rng.uniform(low=-1.0, high=1.0, size=8)
    z0 = rng.normal(scale=0.05, size=8)

    def normal_fn(x: ptv.TensorVariable, y: ptv.TensorVariable, z: ptv.TensorVariable):
        normal = fit_lspl(x, y, z).normal
        return ptb.stack([normal.x, normal.y, normal.z])

    pg.verify_grad(
        normal_fn, [x0, y0, z0], rng=np.random.default_rng(seed=seed + 100)
    )


# pylint: disable=duplicate-code
@pytest.mark.parametrize("x_dim, y_dim, z_dim", utils.NON_1D_SHAPE_CASES)
def test_rejects_non_1d_input_constant(x_dim: int, y_dim: int, z_dim: int) -> None:
    """
    A non-1-dimensional x, y, or z should raise ValueError immediately,
    when given as NumPy arrays (converted internally to TensorConstant).
    """

    with pytest.raises(ValueError, match=MSG_NOT_1D):
        fit_lspl(
            x=np.zeros((3,) * x_dim),
            y=np.zeros((3,) * y_dim),
            z=np.zeros((3,) * z_dim)
        )


@pytest.mark.parametrize("x_dim, y_dim, z_dim", utils.NON_1D_SHAPE_CASES)
def test_rejects_non_1d_input_symbolic(x_dim: int, y_dim: int, z_dim: int) -> None:
    """
    A non-1-dimensional x, y, or z should raise ValueError immediately,
    when given as symbolic (shapeless) PyTensor variables.
    """

    def _make(dim: int, name: str) -> ptv.TensorVariable:

        if dim == 1:
            return ptt.vector(name)
        if dim == 2:
            return ptt.matrix(name)
        if dim == 3:
            return ptt.tensor3(name)

        raise ValueError("`dim` must be less than 4")

    x = _make(dim=x_dim, name="x")
    y = _make(dim=y_dim, name="y")
    z = _make(dim=z_dim, name="z")

    with pytest.raises(ValueError, match=MSG_NOT_1D):
        fit_lspl(x=x, y=y, z=z)


# pylint: disable=duplicate-code
@pytest.mark.parametrize("num_points", range(0, 3))
def test_requires_at_least_three_points_dynamic(num_points: int) -> None:
    """
    When the point count is not statically known, a count below 3
    should not raise at fit() call time, but should raise when the
    graph is evaluated.
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

    with pytest.raises(AssertionError, match=MSG_MIN_POINTS):
        fn(np.zeros(num_points), np.zeros(num_points), np.zeros(num_points))


@pytest.mark.parametrize("num_points", range(0, 3))
def test_requires_at_least_three_points_static(num_points: int) -> None:
    """Reject fewer than three points."""
    with pytest.raises(ValueError, match=MSG_MIN_POINTS):
        fit_lspl(
            x=np.zeros(num_points),
            y=np.zeros(num_points),
            z=np.zeros(num_points)
        )
