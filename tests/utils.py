"""Test utilities."""

import numpy as np
import pytest

from pylspl.result import Vector3D


MISMATCHED_LENGTH_CASES = [
    (3, 4, 4), (4, 3, 4), (4, 4, 3),
    (5, 4, 4), (4, 5, 4), (4, 4, 5),
    (3, 4, 5), (4, 5, 3), (5, 3, 4),
    (5, 4, 3), (4, 3, 5), (3, 5, 4)
]


def make_axis_aligned_coords(
    axis: str, num_points: int, seed: int = 42
) -> dict[str, np.ndarray]:
    """
    Build point coordinates lying on the {axis}=0 plane.

    Parameters
    ----------
    axis
        Which axis ("x", "y", or "z") the plane is flattened along.
    num_points
        Number of points to generate.
    seed
        Seed for the random coordinate generator.

    Returns
    -------
    dict[str, np.ndarray]
        Coordinates keyed by "x", "y", "z".
    """

    rng = np.random.default_rng(seed=seed)

    coords = {
        "x": rng.uniform(low=-1.0, high=1.0, size=num_points),
        "y": rng.uniform(low=-1.0, high=1.0, size=num_points),
        "z": rng.uniform(low=-1.0, high=1.0, size=num_points),
    }

    coords[axis] = np.zeros(num_points)

    return coords


def make_mirrored_points(
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


def make_tilted_plane_coords(
    num_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Vector3D]:
    """
    Build point coordinates on a randomly tilted plane z = a*x + b*y + c.

    Returns
    -------
    tuple
        (x, y, z, desired_normal), where desired_normal is the exact
        unit normal of the plane the points were generated on.
    """

    rng = np.random.default_rng(seed=seed)

    a, b, c = rng.uniform(low=-3.0, high=3.0, size=3)

    desired_normal = Vector3D(x=a, y=b, z=-1.0).normalize()

    x = rng.uniform(low=-1.0, high=1.0, size=num_points)
    y = rng.uniform(low=-1.0, high=1.0, size=num_points)
    z = a * x + b * y + c

    return x, y, z, desired_normal


def assert_normal_aligned_with_axis(normal: dict[str, float], axis: str) -> None:
    """
    Assert that a fitted normal is parallel to the given coordinate axis.

    The sign of the normal is not determined by the fitting algorithm,
    so alignment is checked via absolute value.

    Parameters
    ----------
    normal
        Fitted normal components keyed by "x", "y", "z".
    axis
        The axis the normal is expected to align with.
    """

    for target_axis in "xyz":
        if target_axis == axis:
            assert abs(abs(normal[target_axis]) - 1.0) == pytest.approx(0.0)
        else:
            assert normal[target_axis] == pytest.approx(0.0)
