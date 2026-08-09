"""Tests for the NumPy backend's parallelism."""


import numpy as np
import pytest

import utils

from pylspl.numpy_backend import parallelism


common_rng = np.random.default_rng(seed=42)


@pytest.mark.parametrize("num_points", range(1, 101))
def test_parallelism_known_value(num_points: int) -> None:
    """Parallelism should equal the known range of signed distances."""

    x, y, z = \
        utils.make_random_coords_with_rng(
            rng=common_rng, num_points=num_points, low=-1.0, high=1.0
        )

    expected_parallelism = np.max(z) - np.min(z)

    actual_parallelism_z0 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_Z0)

    actual_parallelism_zp1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_ZP1)
    actual_parallelism_zn1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_ZN1)

    assert actual_parallelism_z0 == pytest.approx(expected_parallelism)

    assert actual_parallelism_zp1 == pytest.approx(expected_parallelism)
    assert actual_parallelism_zn1 == pytest.approx(expected_parallelism)
