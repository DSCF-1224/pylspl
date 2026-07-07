"""Tests for the NumPy backend."""

import numpy as np
import pytest

from pylspl.numpy_backend import fit as fit_lspl


@pytest.mark.parametrize("num_points", range(3, 11))
def test_fit_exact_plane(num_points: int) -> None:
    """
    For points on the z=0 plane, the normal should point along z and the flatness should be 0.
    """

    rng = np.random.default_rng(seed=42)

    x = rng.uniform(low=-1.0, high=1.0, size=num_points)
    y = rng.uniform(low=-1.0, high=1.0, size=num_points)
    z = np.zeros_like(x)

    result = fit_lspl(x=x, y=y, z=z)

    # The normal should be parallel to the z axis (sign is undefined)
    assert abs(abs(result.normal.z) - 1.0) < 1e-10
    assert result.normal.x == pytest.approx(0.0)
    assert result.normal.y == pytest.approx(0.0)

    # The flatness should be zero
    assert result.flatness == pytest.approx(0.0)
