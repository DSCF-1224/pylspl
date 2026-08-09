"""Tests for the NumPy backend's parallelism."""


import numpy as np
import pytest

import utils

from pylspl.numpy_backend import parallelism


common_rng = np.random.default_rng(seed=42)


# pylint: disable=too-many-locals
@pytest.mark.parametrize("num_points", range(1, 101))
def test_parallelism_known_value(num_points: int) -> None:
    """Parallelism should equal the known range of signed distances."""

    x, y, z = \
        utils.make_random_coords_with_rng(
            rng=common_rng, num_points=num_points, low=-1.0, high=1.0
        )

    expected_parallelism_x = np.max(x) - np.min(x)
    expected_parallelism_y = np.max(y) - np.min(y)
    expected_parallelism_z = np.max(z) - np.min(z)

    actual_parallelism_x0 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_X0)
    actual_parallelism_y0 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_Y0)
    actual_parallelism_z0 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_Z0)

    actual_parallelism_xp1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_XP1)
    actual_parallelism_xn1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_XN1)

    actual_parallelism_yp1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_YP1)
    actual_parallelism_yn1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_YN1)

    actual_parallelism_zp1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_ZP1)
    actual_parallelism_zn1 = parallelism(x=x, y=y, z=z, datum=utils.PLANE_ZN1)

    assert actual_parallelism_x0 == pytest.approx(expected_parallelism_x)
    assert actual_parallelism_y0 == pytest.approx(expected_parallelism_y)
    assert actual_parallelism_z0 == pytest.approx(expected_parallelism_z)

    assert actual_parallelism_xp1 == pytest.approx(expected_parallelism_x)
    assert actual_parallelism_xn1 == pytest.approx(expected_parallelism_x)

    assert actual_parallelism_yp1 == pytest.approx(expected_parallelism_y)
    assert actual_parallelism_yn1 == pytest.approx(expected_parallelism_y)

    assert actual_parallelism_zp1 == pytest.approx(expected_parallelism_z)
    assert actual_parallelism_zn1 == pytest.approx(expected_parallelism_z)
