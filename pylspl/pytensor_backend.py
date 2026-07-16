"""PyTensor implementation of LSPL plane fitting."""

from typing import cast

import pytensor.raise_op as pr
import pytensor.tensor as pt
import pytensor.tensor.basic as ptb
import pytensor.tensor.linalg as ptl
import pytensor.tensor.math as ptm
import pytensor.tensor.variable as ptv

from .result import FittedPlane3D, Vector3D


_MSG_MIN_POINTS = "at least 3 points are required"
_MSG_SAME_LENGTH = "x, y, and z must have the same length"

_assert_min_points = pr.Assert(_MSG_MIN_POINTS)
_assert_same_length = pr.Assert(_MSG_SAME_LENGTH)


def _construct_covariance_matrix(
    x: pt.TensorLike,
    y: pt.TensorLike,
    z: pt.TensorLike
) -> ptv.TensorVariable:
    """
    Construct the covariance matrix from centroid-centered coordinates.

    Parameters
    ----------
    x, y, z
        Centroid-centered coordinates.

    Returns
    -------
    TensorVariable
        A 3x3 symmetric matrix.
    """

    xx = ptm.sum(ptm.mul(x, x))
    xy = ptm.sum(ptm.mul(x, y))
    xz = ptm.sum(ptm.mul(x, z))
    yy = ptm.sum(ptm.mul(y, y))
    yz = ptm.sum(ptm.mul(y, z))
    zz = ptm.sum(ptm.mul(z, z))

    return cast(
        ptv.TensorVariable,
        ptb.stack(
            [
                ptb.stack([xx, xy, xz]),
                ptb.stack([xy, yy, yz]),
                ptb.stack([xz, yz, zz])
            ]
        )
    )


def _validate_xyz_shapes(
        x: ptv.TensorVariable,
        y: ptv.TensorVariable,
        z: ptv.TensorVariable
) -> tuple[ptv.TensorVariable, ptv.TensorVariable, ptv.TensorVariable]:
    """
    Validate x, y, z shapes.

    x, y, and z must each be 1-dimensional; since ndim is always
    statically known in PyTensor, this is checked unconditionally.
    Length mismatches and point-count violations raise ValueError
    immediately when shapes are statically known; otherwise the
    checks are embedded in the computation graph and only raise
    when the graph is evaluated (e.g. via `.eval()` or a compiled
    function).

    Raises
    ------
    ValueError
        If x, y, or z is not 1-dimensional, if x, y, and z have
        different lengths, or if fewer than three points are
        provided (the latter two only when shapes are statically
        known).
    AssertionError
        If x, y, and z have different lengths and this could not
        be determined statically (raised when the graph is
        evaluated, via the embedded pytensor.raise_op.Assert
        check).
    """

    if x.type.ndim != 1 or y.type.ndim != 1 or z.type.ndim != 1:
        raise ValueError("x, y, and z must be 1-dimensional")

    size_x = x.type.shape[0]
    size_y = y.type.shape[0]
    size_z = z.type.shape[0]

    if (size_x is not None) and (size_y is not None) and (size_z is not None):

        if size_x != size_y or size_x != size_z:
            raise ValueError(_MSG_SAME_LENGTH)

        if size_x < 3:
            raise ValueError(_MSG_MIN_POINTS)

        return x, y, z

    length_x = x.shape[0]

    x_checked = cast(
        ptv.TensorVariable,
        _assert_same_length(
            x,
            ptm.eq(length_x, y.shape[0]),
            ptm.eq(length_x, z.shape[0]),
        )
    )

    x_checked = cast(
        ptv.TensorVariable,
        _assert_min_points(x_checked, length_x >= 3)
    )

    return x_checked, y, z


def fit(x: pt.TensorLike, y: pt.TensorLike, z: pt.TensorLike) -> FittedPlane3D:
    """
    Fit a least-squares reference plane (orthogonal-distance minimization)
    from a set of points.

    Parameters
    ----------
    x, y, z
        Point coordinates.

    Returns
    -------
    FittedPlane3D
        Fitted plane and evaluated flatness.

    Raises
    ------
    ValueError
        If x, y, and z have different lengths or fewer than
        three points are provided (only when shapes are
        statically known).
    AssertionError
        If x, y, and z have different lengths and this could not
        be determined statically (raised when the graph is
        evaluated, via the embedded pytensor.raise_op.Assert
        check).
    LinAlgError
        If the eigendecomposition does not converge. Because PyTensor
        builds a symbolic computation graph, this error is raised when
        the graph is evaluated (e.g. via `.eval()` or a compiled function),
        rather than when `fit` is called.
    """

    x_tensor, y_tensor, z_tensor = _validate_xyz_shapes(
        x=ptb.as_tensor_variable(x),
        y=ptb.as_tensor_variable(y),
        z=ptb.as_tensor_variable(z)
    )

    centroid = Vector3D(x=ptm.mean(x_tensor), y=ptm.mean(
        y_tensor), z=ptm.mean(z_tensor))

    x_offset = x_tensor - centroid.x
    y_offset = y_tensor - centroid.y
    z_offset = z_tensor - centroid.z

    _, eigenvectors = cast(
        tuple[ptv.TensorVariable, ptv.TensorVariable],
        ptl.eigh(_construct_covariance_matrix(x_offset, y_offset, z_offset))
    )

    # eigenvector for the smallest eigenvalue
    normal = Vector3D(
        x=eigenvectors[0, 0],
        y=eigenvectors[1, 0],
        z=eigenvectors[2, 0]
    )

    distances = (
        (x_offset * normal.x) +
        (y_offset * normal.y) +
        (z_offset * normal.z)
    )

    return FittedPlane3D(
        point=centroid,
        normal=normal,
        flatness=ptm.max(distances) - ptm.min(distances),
    )
