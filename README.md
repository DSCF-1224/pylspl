# pylspl

[![CI](https://github.com/DSCF-1224/pylspl/actions/workflows/ci.yml/badge.svg)](https://github.com/DSCF-1224/pylspl/actions/workflows/ci.yml)

Python implementation of Least Squares Reference Plane (LSPL) fitting and flatness evaluation for 3D points

## Installation

`pylspl` is not published to PyPI. Install directly from this repository.

### pip

```bash
pip install "git+https://github.com/DSCF-1224/pylspl.git"
```

### uv

```bash
uv add "git+https://github.com/DSCF-1224/pylspl.git"
```

## Usage

### `NumPy` backend

```python
import numpy as np
from pylspl.numpy_backend import fit

x = np.array([0.0, 1.0, 0.0, 1.0])
y = np.array([0.0, 0.0, 1.0, 1.0])
z = np.array([0.0, 0.1, -0.1, 0.05])

result = fit(x=x, y=y, z=z)
print(result.point, result.normal, result.flatness)
```

### `PyTensor` backend

```python
import numpy as np
from pylspl.pytensor_backend import fit

x = np.array([0.0, 1.0, 0.0, 1.0])
y = np.array([0.0, 0.0, 1.0, 1.0])
z = np.array([0.0, 0.1, -0.1, 0.05])

result = fit(x=x, y=y, z=z)

print(result.point.x.eval())
print(result.point.y.eval())
print(result.point.z.eval())

print(result.normal.x.eval())
print(result.normal.y.eval())
print(result.normal.z.eval())

print(result.flatness.eval())
```

`point`, `normal`, and `flatness` are symbolic ([`TensorVariable`](https://pytensor.readthedocs.io/en/stable/library/tensor/basic.html#pytensor.tensor.TensorVariable));
call [`.eval()`](https://pytensor.readthedocs.io/en/stable/library/graph/graph.html#pytensor.graph.basic.Variable.eval) to obtain numeric values, or use [`pytensor.function`](https://pytensor.readthedocs.io/en/stable/library/compile/function.html#pytensor.compile.maker.function) to compile
a reusable function.

Inputs must be 1-dimensional. See [Algorithm documentation](docs/algorithm.md#backend-differences)
for the PyTensor backend's automatic differentiation support and its
limitation for point sets with (near-)repeated covariance-matrix eigenvalues.

## Algorithm

See [Algorithm documentation](docs/algorithm.md)
for the mathematical derivation and implementation details.
