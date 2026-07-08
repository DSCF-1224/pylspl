"""Tests for the NumPy backend."""

import numpy as np
import pytest

from pylspl.numpy_backend import fit as fit_lspl


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
