"""NumPy implementation of LSPL plane fitting."""

from typing import Any

import numpy as np

from ._messages import MSG_MIN_POINTS, MSG_NOT_1D, MSG_SAME_LENGTH
from .result import FittedPlane3D, Plane3D, Vector3D


def _construct_covariance_matrix(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Construct the covariance matrix from centroid-centered coordinates.

    Parameters
    ----------
    x, y, z
        Centroid-centered coordinates.

    Returns
    -------
    np.ndarray
        A 3x3 symmetric matrix.
    """

    xx = np.sum(x * x)
    xy = np.sum(x * y)
    xz = np.sum(x * z)
    yy = np.sum(y * y)
    yz = np.sum(y * z)
    zz = np.sum(z * z)

    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _validate_xyz_ndim(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    Validate x, y, z ndim.

    Raises
    ------
    ValueError
        If x, y, or z is not 1-dimensional.
    """

    if np.ndim(x) != 1 or np.ndim(y) != 1 or np.ndim(z) != 1:
        raise ValueError(MSG_NOT_1D)


def _validate_xyz_size(size_x: int, size_y: int, size_z: int) -> None:
    """
    Validate x, y, z size.

    Raises
    ------
    ValueError
        If x, y, and z have different lengths.
    """

    if size_x != size_y or size_x != size_z:
        raise ValueError(MSG_SAME_LENGTH)


def fit(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> FittedPlane3D:
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
        If x, y, or z is not 1-dimensional, if x, y, and z have
        different lengths, or if fewer than three points are
        provided.
    numpy.linalg.LinAlgError
        If the eigendecomposition does not converge.
    """

    _validate_xyz_ndim(x=x, y=y, z=z)

    size_x = np.size(x)

    _validate_xyz_size(size_x=size_x, size_y=np.size(y), size_z=np.size(z))

    if size_x < 3:
        raise ValueError(MSG_MIN_POINTS)

    centroid = Vector3D(x=np.mean(x), y=np.mean(y), z=np.mean(z))

    x_offset = x - centroid.x
    y_offset = y - centroid.y
    z_offset = z - centroid.z

    matrix = _construct_covariance_matrix(x_offset, y_offset, z_offset)

    _, eigenvectors = np.linalg.eigh(matrix)

    # eigenvector for the smallest eigenvalue
    normal = Vector3D(x=eigenvectors[0, 0],
                      y=eigenvectors[1, 0],
                      z=eigenvectors[2, 0])

    distances = \
        (x_offset * normal.x) + \
        (y_offset * normal.y) + \
        (z_offset * normal.z)

    return FittedPlane3D(
        point=centroid,
        normal=normal,
        flatness=np.max(distances) - np.min(distances),
    )


def parallelism(x: np.ndarray, y: np.ndarray, z: np.ndarray, datum: Plane3D) -> Any:
    """
    Evaluate the parallelism of a point set with respect to a datum plane.

    For each point, the signed distance to the datum plane (measured
    along the datum's normal direction) is computed. The result is
    the range (max - min) of these signed distances: zero when the
    point set lies exactly on a plane parallel to the datum, larger
    as the point set deviates from that orientation.

    Parameters
    ----------
    x, y, z
        Point coordinates.
    datum
        The reference (datum) plane. Its normal need not be unit
        length; it is normalized internally.

    Returns
    -------
    Any
        The parallelism value.

    Raises
    ------
    ValueError
        If x, y, or z is not 1-dimensional, or if x, y, and z have
        different lengths.
    """

    _validate_xyz_ndim(x=x, y=y, z=z)

    _validate_xyz_size(size_x=np.size(x), size_y=np.size(y), size_z=np.size(z))

    datum_normal = datum.normal.normalize()

    dx = x - datum.point.x
    dy = y - datum.point.y
    dz = z - datum.point.z

    distances = (dx * datum_normal.x) + \
        (dy * datum_normal.y) + (dz * datum_normal.z)

    return np.max(distances) - np.min(distances)
