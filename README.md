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

Not yet implemented.

## Algorithm

See [Algorithm documentation](docs/algorithm.md)
for the mathematical derivation and implementation details.
