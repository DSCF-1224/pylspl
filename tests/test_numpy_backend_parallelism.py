"""Tests for the NumPy backend's parallelism."""


import numpy as np
import pytest

import utils

from pylspl.numpy_backend import parallelism
from pylspl.result import Plane3D, Vector3D
from pylspl._messages import MSG_NOT_1D


common_rng = np.random.default_rng(seed=42)


# pylint: disable=too-many-locals
@pytest.mark.parametrize("num_points", range(1, 101))
def test_parallelism_datum_normal_need_not_be_unit(num_points: int) -> None:
    """A non-unit-length datum normal should be normalized internally."""

    x, y, z = \
        utils.make_random_coords_with_rng(
            rng=common_rng, num_points=num_points, low=-1.0, high=1.0
        )

    scaled_datum_xp5 = Plane3D(
        point=Vector3D(x=0.0, y=0.0, z=0.0),
        normal=utils.NORMAL_VECTOR_X_AXIS * 5
    )

    scaled_datum_xn5 = Plane3D(
        point=Vector3D(x=0.0, y=0.0, z=0.0),
        normal=utils.NORMAL_VECTOR_X_AXIS * (-5)
    )

    scaled_datum_yp5 = Plane3D(
        point=Vector3D(x=0.0, y=0.0, z=0.0),
        normal=utils.NORMAL_VECTOR_Y_AXIS * 5
    )

    scaled_datum_yn5 = Plane3D(
        point=Vector3D(x=0.0, y=0.0, z=0.0),
        normal=utils.NORMAL_VECTOR_Y_AXIS * (-5)
    )

    scaled_datum_zp5 = Plane3D(
        point=Vector3D(x=0.0, y=0.0, z=0.0),
        normal=utils.NORMAL_VECTOR_Z_AXIS * 5
    )

    scaled_datum_zn5 = Plane3D(
        point=Vector3D(x=0.0, y=0.0, z=0.0),
        normal=utils.NORMAL_VECTOR_Z_AXIS * (-5)
    )

    expected_parallelism_x = parallelism(x=x, y=y, z=z, datum=utils.PLANE_X0)
    expected_parallelism_y = parallelism(x=x, y=y, z=z, datum=utils.PLANE_Y0)
    expected_parallelism_z = parallelism(x=x, y=y, z=z, datum=utils.PLANE_Z0)

    actual_parallelism_xp5 = parallelism(x=x, y=y, z=z, datum=scaled_datum_xp5)
    actual_parallelism_xn5 = parallelism(x=x, y=y, z=z, datum=scaled_datum_xn5)

    actual_parallelism_yp5 = parallelism(x=x, y=y, z=z, datum=scaled_datum_yp5)
    actual_parallelism_yn5 = parallelism(x=x, y=y, z=z, datum=scaled_datum_yn5)

    actual_parallelism_zp5 = parallelism(x=x, y=y, z=z, datum=scaled_datum_zp5)
    actual_parallelism_zn5 = parallelism(x=x, y=y, z=z, datum=scaled_datum_zn5)

    assert actual_parallelism_xp5 == pytest.approx(expected_parallelism_x)
    assert actual_parallelism_xn5 == pytest.approx(expected_parallelism_x)

    assert actual_parallelism_yp5 == pytest.approx(expected_parallelism_y)
    assert actual_parallelism_yn5 == pytest.approx(expected_parallelism_y)

    assert actual_parallelism_zp5 == pytest.approx(expected_parallelism_z)
    assert actual_parallelism_zn5 == pytest.approx(expected_parallelism_z)


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


# pylint: disable=duplicate-code
@pytest.mark.parametrize("x_dim, y_dim, z_dim", utils.NON_1D_SHAPE_CASES)
def test_parallelism_rejects_non_1d_input(x_dim: int, y_dim: int, z_dim: int) -> None:
    """
    A non-1-dimensional x, y, or z should raise ValueError immediately,
    regardless of the datum plane.
    """

    for plane in [utils.PLANE_X0, utils.PLANE_Y0, utils.PLANE_Z0]:
        with pytest.raises(ValueError, match=MSG_NOT_1D):
            parallelism(
                x=np.zeros((3,) * x_dim),
                y=np.zeros((3,) * y_dim),
                z=np.zeros((3,) * z_dim),
                datum=plane
            )
