"""Tests for consistency between the NumPy and PyTensor backends."""

import numpy as np
import pytensor
import pytest

from pylspl import numpy_backend, pytensor_backend
from pylspl.result import Vector3D


@pytest.mark.parametrize("num_points", range(3, 101))
def test_fit_produces_consistent_results(num_points: int) -> None:
    """Ensure that both backends produce equivalent fitting results."""

    rng = np.random.default_rng(42)

    x = rng.uniform(low=-1.0, high=1.0, size=num_points)
    y = rng.uniform(low=-1.0, high=1.0, size=num_points)
    z = rng.uniform(low=-1.0, high=1.0, size=num_points)

    np_result = numpy_backend.fit(x=x, y=y, z=z)
    pt_result = pytensor_backend.fit(x=x, y=y, z=z)

    pt_point_x, pt_point_y, pt_point_z, pt_normal_x, pt_normal_y, pt_normal_z, pt_flatness = \
        pytensor.function(  # pyright: ignore[reportPrivateImportUsage]
            [],
            [pt_result.point.x, pt_result.point.y, pt_result.point.z,
             pt_result.normal.x, pt_result.normal.y, pt_result.normal.z,
             pt_result.flatness]
        )()

    pt_normal = Vector3D(x=pt_normal_x, y=pt_normal_y, z=pt_normal_z)

    assert pt_point_x == pytest.approx(np_result.point.x)
    assert pt_point_y == pytest.approx(np_result.point.y)
    assert pt_point_z == pytest.approx(np_result.point.z)

    assert abs(pt_normal.dot(np_result.normal)) == pytest.approx(1.0)

    assert pt_flatness == pytest.approx(np_result.flatness)
