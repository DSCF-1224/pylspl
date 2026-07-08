"""Tests for the NumPy backend."""

import numpy as np
import pytest

from pylspl.numpy_backend import fit as fit_lspl
from pylspl.result import Vector3D


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
