"""PyTensor implementation of LSPL plane fitting."""

from typing import cast

import pytensor.tensor as pt
import pytensor.tensor.basic as ptb
import pytensor.tensor.linalg as ptl
import pytensor.tensor.math as ptm
import pytensor.tensor.variable as ptv

from .result import FittedPlane3D, Vector3D


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


def _validate_lengths(x: ptv.TensorVariable, y: ptv.TensorVariable, z: ptv.TensorVariable) -> None:
    """
    Validate x, y, z lengths when statically known.

    Raises
    ------
    ValueError
        If x, y, and z have different lengths or fewer than
        three points are provided (only when shapes are
        statically known).
    """

    size_x = x.type.shape[0]
    size_y = y.type.shape[0]
    size_z = z.type.shape[0]

    if (size_x is not None) and (size_y is not None) and (size_z is not None):
        if size_x != size_y or size_x != size_z:
            raise ValueError("x, y, and z must have the same length")
        if size_x < 3:
            raise ValueError("at least 3 points are required")


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
    LinAlgError
        If the eigendecomposition does not converge. Because PyTensor
        builds a symbolic computation graph, this error is raised when
        the graph is evaluated (e.g. via `.eval()` or a compiled function),
        rather than when `fit` is called.
    """

    x_tensor = ptb.as_tensor_variable(x)
    y_tensor = ptb.as_tensor_variable(y)
    z_tensor = ptb.as_tensor_variable(z)

    _validate_lengths(x=x_tensor, y=y_tensor, z=z_tensor)

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
