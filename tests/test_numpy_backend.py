"""Tests for the NumPy backend."""

import numpy as np
import pytest

from pylspl.numpy_backend import fit as fit_lspl
from pylspl.result import Vector3D


def _make_mirrored_points(
    rng: np.random.Generator, num_base_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate a point set with exactly known flatness and orientation.

    Each of `num_base_points` randomly placed (x, y) points is mirrored
    to z = +delta and z = -delta, which makes the xz/yz cross-covariance
    terms vanish exactly regardless of the (x, y) values. The point set
    is then rotated by a random rotation matrix.

    Returns
    -------
    tuple
        (x, y, z, delta) where 2 * delta is the exact expected flatness.
    """

    x0 = rng.uniform(low=-1.0, high=1.0, size=num_base_points)
    y0 = rng.uniform(low=-1.0, high=1.0, size=num_base_points)

    xc = x0 - np.mean(x0)
    yc = y0 - np.mean(y0)

    inplane_cov = np.array(
        [[np.sum(xc * xc), np.sum(xc * yc)],
         [np.sum(xc * yc), np.sum(yc * yc)]]
    )

    lambda_min_inplane = 2 * np.linalg.eigvalsh(inplane_cov)[0]

    delta = rng.uniform(low=0.05, high=0.5) \
        * np.sqrt(lambda_min_inplane / (2 * num_base_points))

    local_x = np.concatenate([x0, x0])
    local_y = np.concatenate([y0, y0])
    local_z = np.concatenate(
        [np.full(num_base_points, delta), np.full(num_base_points, -delta)]
    )

    x, y, z = _make_rotation(rng) @ np.stack([local_x, local_y, local_z])

    return x, y, z, delta


def _make_rotation(rng: np.random.Generator) -> np.ndarray:
    """Return a uniformly random 3x3 orthonormal (rotation) matrix."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("num_points", range(3, 11))
def test_fit_exact_plane(axis: str, num_points: int) -> None:
    """
    For points on the {axis}=0 plane,
    the normal should point along {axis} and the flatness should be 0.
    """

    rng = np.random.default_rng(seed=42)

    coords = {
        "x": rng.uniform(low=-1.0, high=1.0, size=num_points),
        "y": rng.uniform(low=-1.0, high=1.0, size=num_points),
        "z": rng.uniform(low=-1.0, high=1.0, size=num_points),
    }

    coords[axis] = np.zeros(num_points)

    result = fit_lspl(x=coords["x"], y=coords["y"], z=coords["z"])

    normal = {"x": result.normal.x, "y": result.normal.y, "z": result.normal.z}

    # The normal should be parallel to the {axis} axis (sign is undefined)
    for target_axis in "xyz":
        if target_axis == axis:
            assert abs(abs(normal[target_axis]) - 1.0) == pytest.approx(0.0)
        else:
            assert normal[target_axis] == pytest.approx(0.0)

    # The flatness should be zero
    assert result.flatness == pytest.approx(0.0)


@pytest.mark.parametrize("num_points", range(3, 11))
@pytest.mark.parametrize("seed", range(0, 10))
def test_fit_tilted_plane(num_points: int, seed: int) -> None:
    """
    A randomly oriented plane should be fitted correctly.
    """

    rng = np.random.default_rng(seed=seed)

    # random plane coefficients: z = a*x + b*y + c
    a, b, c = rng.uniform(low=-3.0, high=3.0, size=3)

    # desired_normal needs normalizing before comparison.
    # (result.normal will be unit norm (eigenvector))
    desired_normal = Vector3D(x=a, y=b, z=-1.0).normalize()

    x = rng.uniform(low=-1.0, high=1.0, size=num_points)
    y = rng.uniform(low=-1.0, high=1.0, size=num_points)
    z = a * x + b * y + c

    result = fit_lspl(x=x, y=y, z=z)

    assert abs(desired_normal.dot(result.normal)) == pytest.approx(1.0)

    # The flatness should be zero
    assert result.flatness == pytest.approx(0.0)


@pytest.mark.parametrize("num_base_points", range(3, 11))
@pytest.mark.parametrize("seed", range(10))
def test_flatness_matches_known_value(seed: int, num_base_points: int) -> None:
    """
    For a point set whose exact flatness is known by construction,
    the fitted flatness should match.
    """

    rng = np.random.default_rng(seed=seed)

    x, y, z, delta = _make_mirrored_points(rng, num_base_points)

    assert fit_lspl(x=x, y=y, z=z).flatness == pytest.approx(2 * delta)


@pytest.mark.parametrize(
    "x_len, y_len, z_len",
    [
        (3, 4, 4), (4, 3, 4), (4, 4, 3),
        (5, 4, 4), (4, 5, 4), (4, 4, 5),
        (3, 4, 5), (4, 5, 3), (5, 3, 4),
        (5, 4, 3), (4, 3, 5), (3, 5, 4)
    ],
)
def test_mismatched_length(x_len: int, y_len: int, z_len: int) -> None:
    """Reject points with mismatched coordinate lengths."""
    with pytest.raises(ValueError, match="must have the same length"):
        fit_lspl(x=np.zeros(x_len), y=np.zeros(y_len), z=np.zeros(z_len))


@pytest.mark.parametrize("num_points", range(0, 3))
def test_requires_at_least_three_points(num_points: int) -> None:
    """Reject fewer than three points."""
    with pytest.raises(ValueError, match="at least 3 points are required"):
        fit_lspl(
            x=np.zeros(num_points),
            y=np.zeros(num_points),
            z=np.zeros(num_points)
        )
