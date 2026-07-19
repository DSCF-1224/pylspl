# Algorithm

## Overview

`pylspl` computes a **Least Squares Reference Plane (LSPL)** from a set of three-dimensional points and evaluates the corresponding flatness.

Given input points

$$
x=(x_1,\ldots,x_n),\qquad
y=(y_1,\ldots,y_n),\qquad
z=(z_1,\ldots,z_n),
$$

`pylspl` computes

- fitted plane point (a point lying on the fitted plane)
- fitted plane unit normal vector
- flatness

using an orthogonal-distance least-squares formulation
and returns a `FittedPlane3D` object.

## Mathematical formulation

### Plane equation

A plane is represented by a point on the plane
$\mathbf{p}_0 = (a,b,c)$
and a unit normal vector
$\mathbf{n} = (n_x,n_y,n_z)$, $\lVert \mathbf{n} \rVert = 1$.

The signed distance from a point $\mathbf{p}=(x,y,z)$ to the plane is

$$
\Delta(\mathbf{p}) := (\mathbf{p}-\mathbf{p}_0)\cdot\mathbf{n}
= (x-a)n_x + (y-b)n_y + (z-c)n_z.
$$

## Least-squares formulation

For measured data, the points generally do not lie exactly on a plane.
For each point,

$$
\Delta_i := (x_i-a)n_x + (y_i-b)n_y + (z_i-c)n_z
$$

is generally nonzero. This is the orthogonal distance from the point to the plane.

The least-squares problem is to find

$$
\mathop{\text{argmin}}\limits_{\mathbf{p}_0,\ \mathbf{n}:\ \lVert \mathbf{n} \rVert=1} \sum_{i=1}^n {(\Delta_i)}^2.
$$

### Step 1 — Optimal plane point

Differentiating the objective with respect to $a$,

```math
\begin{aligned}
\frac{\partial}{\partial a}\sum_{i=1}^n {(\Delta_i)}^2
&= \sum_{i=1}^n \frac{\partial}{\partial a} {(\Delta_i)}^2 \\
&= \sum_{i=1}^n (2\Delta_i) \cdot \frac{\partial \Delta_i}{\partial a} \\
&= \sum_{i=1}^n (2\Delta_i) \cdot \frac{\partial}{\partial a} \bigl\lbrace (x_i-a)n_x + (y_i-b)n_y + (z_i-c)n_z \bigr\rbrace \\
&= -2n_x \sum_{i=1}^n \Delta_i.
\end{aligned}
```

By the symmetry of $\Delta_i$ under the cyclic substitution $(x,a,n_x)\to(y,b,n_y)\to(z,c,n_z)$, the partial derivatives with respect to $b$ and $c$ take the same form:

$$
\frac{\partial}{\partial b}\sum_{i=1}^n {(\Delta_i)}^2 = -2n_y\sum_{i=1}^n \Delta_i,\qquad
\frac{\partial}{\partial c}\sum_{i=1}^n {(\Delta_i)}^2 = -2n_z\sum_{i=1}^n \Delta_i.
$$

Setting the gradient to zero, and noting $\mathbf{n}\neq\mathbf{0}$ (since $\lVert\mathbf{n}\rVert=1$), all three equations reduce to a single scalar condition:

$$
\sum_{i=1}^n \Delta_i = 0.
$$

Expanding,

$$
\sum_{i=1}^n \Delta_i
= n_x\left\lbrace \left(\sum_{i=1}^n x_i\right) - na\right\rbrace
+ n_y\left\lbrace \left(\sum_{i=1}^n y_i\right) - nb\right\rbrace
+ n_z\left\lbrace \left(\sum_{i=1}^n z_i\right) - nc\right\rbrace = 0.
$$

This is a single scalar equation in the three unknowns $a,b,c$ (for fixed $\mathbf{n}$); it does not by itself pin down $a,b,c$ individually — only the component of $\mathbf{p}_0=(a,b,c)$ along $\mathbf{n}$. This reflects a genuine indeterminacy in the problem: since $\Delta_i$ depends on $\mathbf{p}_0$ only through $\mathbf{n}\cdot\mathbf{p}_0$, sliding $\mathbf{p}_0$ within the plane (i.e., by any vector orthogonal to $\mathbf{n}$) leaves every $\Delta_i$, and hence the fitted plane itself, unchanged. The condition above constrains only the projection of $\mathbf{p}_0$ onto $\mathbf{n}$:

$$
\mathbf{n} \cdot \mathbf{p}_0 = \mathbf{n} \cdot \bar{\mathbf{p}},
\qquad
\bar{\mathbf{p}} := \left(\bar{x},\bar{y},\bar{z}\right),
\qquad
\bar{x}:=\frac{1}{n}\sum_{i=1}^n x_i,\ \ \bar{y}:=\frac{1}{n}\sum_{i=1}^n y_i,\ \ \bar{z}:=\frac{1}{n}\sum_{i=1}^n z_i.
$$

The centroid $\bar{\mathbf{p}}$ trivially satisfies this condition, and adopting it as $\mathbf{p}_0$ removes the remaining ambiguity in the components perpendicular to $\mathbf{n}$. The implementation therefore uses

$$
\mathbf{p}_0 = \bar{\mathbf{p}} = (\bar{x},\bar{y},\bar{z}).
$$

### Step 2 — Optimal normal vector

With $\mathbf{p}_0$ fixed at the centroid, introduce offset coordinates

$$
x_i^\prime := x_i-\bar{x},\qquad
y_i^\prime := y_i-\bar{y},\qquad
z_i^\prime := z_i-\bar{z}.
$$

Then $\Delta_i = x_i^\prime n_x + y_i^\prime n_y + z_i^\prime n_z$, and the objective becomes a quadratic form,

$$
\sum_{i=1}^n {(\Delta_i)}^2 = \mathbf{n}^{\top} M \mathbf{n},
\qquad
M := \sum_{i=1}^n
\begin{bmatrix}
{(x_i^\prime)}^2 & x_i^\prime y_i^\prime & x_i^\prime z_i^\prime \\
x_i^\prime y_i^\prime & {(y_i^\prime)}^2 & y_i^\prime z_i^\prime \\
x_i^\prime z_i^\prime & y_i^\prime z_i^\prime & {(z_i^\prime)}^2
\end{bmatrix}.
$$

$M$ is the (unnormalized) covariance matrix of the centered points; it is real and symmetric.

#### Why the smallest eigenvalue's eigenvector minimizes the objective

Because $M$ is real symmetric, the spectral theorem guarantees an orthonormal basis of eigenvectors
$\mathbf{e}_1,\mathbf{e}_2,\mathbf{e}_3$ with corresponding real eigenvalues
$\lambda_1\le\lambda_2\le\lambda_3$, i.e. $M\mathbf{e}_k=\lambda_k\mathbf{e}_k$.

Any unit vector $\mathbf{n}$ can be written in this basis as

$$
\mathbf{n} = c_1\mathbf{e}_1 + c_2\mathbf{e}_2 + c_3\mathbf{e}_3,
\qquad
c_1^2+c_2^2+c_3^2 = \lVert\mathbf{n}\rVert^2 = 1,
$$

since the basis is orthonormal. Substituting into the objective,

$$
\mathbf{n}^{\top} M \mathbf{n}
= \sum_{k=1}^3 c_k^2\, \mathbf{e}_k^{\top} M \mathbf{e}_k
= \sum_{k=1}^3 c_k^2 \lambda_k,
$$

using $\mathbf{e}_j^{\top}M\mathbf{e}_k=\lambda_k\mathbf{e}_j^{\top}\mathbf{e}_k=\lambda_k\delta_{jk}$.

This shows $\mathbf{n}^{\top}M\mathbf{n}$ is a **weighted average of the eigenvalues**, with weights $c_k^2$ summing to 1. A weighted average of a set of numbers is always bounded below by their minimum, with equality exactly when all weight is placed on the minimizing term:

$$
\mathbf{n}^{\top} M \mathbf{n} = \sum_{k=1}^3 c_k^2 \lambda_k \ \ge\ \lambda_{\min}\sum_{k=1}^3 c_k^2 = \lambda_{\min},
$$

with equality if and only if $c_k=0$ for every $k$ with $\lambda_k>\lambda_{\min}$ — that is, if and only if $\mathbf{n}$ lies entirely along the eigenspace of $\lambda_{\min}$.

Hence the minimum of $\mathbf{n}^{\top}M\mathbf{n}$ over all unit vectors $\mathbf{n}$ is exactly $\lambda_{\min}$, the smallest eigenvalue of $M$, and it is attained by (any) unit eigenvector belonging to $\lambda_{\min}$:

$$
\min_{\lVert\mathbf{n}\rVert=1} \mathbf{n}^{\top}M\mathbf{n} = \lambda_{\min},
\qquad
\mathbf{n}^{\star} = \mathbf{e}_{\arg\min_k \lambda_k}.
$$

(For reference, the same conclusion follows from a Lagrange-multiplier stationarity argument: setting $\nabla_{\mathbf{n}}\left[\mathbf{n}^{\top}M\mathbf{n}-\lambda(\mathbf{n}^{\top}\mathbf{n}-1)\right]=\mathbf{0}$ gives $M\mathbf{n}=\lambda\mathbf{n}$, so every critical point is some eigenvector $\mathbf{e}_k$ with objective value $\lambda_k$; the spectral argument above additionally confirms that among these critical points, $\lambda_{\min}$ is the *global* minimum rather than merely a stationary value.)

This is exactly the eigendecomposition performed in the implementation.

### Solving the eigenvalue problem

The resulting $3\times3$ symmetric eigenvalue problem is solved using

* [`numpy.linalg.eigh`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)
* [`pytensor.tensor.linalg.eigh`](https://pytensor.readthedocs.io/en/stable/library/tensor/linalg.html#pytensor.tensor.linalg.eigh)

depending on the selected backend. See [Backend differences](#backend-differences)
for the PyTensor backend's differentiability and its limitation for
point sets with (near-)repeated covariance-matrix eigenvalues.

## Recovering the plane

$$
\mathbf{p}_0 = (\bar{x},\bar{y},\bar{z}),\qquad
\mathbf{n} = \text{eigenvector of } M \text{ for } \lambda_{\min}.
$$

The sign of $\mathbf{n}$ is not determined by this formulation; both $\mathbf{n}$ and $-\mathbf{n}$ describe the same plane.

## Flatness evaluation

For each measured point,

$$
\delta_i := \Delta_i = (\mathbf{p}_i - \mathbf{p}_0)\cdot\mathbf{n}
$$

Flatness is defined as

$$
\max_i \delta_i - \min_i \delta_i.
$$

Consequently,

- a perfect plane yields zero flatness,
- deviations from planarity increase the value.

Because $\mathbf{n}$'s sign is unresolved, $\delta_i$ may flip sign uniformly, but $\max_i \delta_i - \min_i \delta_i$ is invariant under $\mathbf{n}\to-\mathbf{n}$.

## Numerical stabilization

Centering the data at the centroid, as derived in [Step 1](#step-1--optimal-plane-point), is a necessary condition of the least-squares solution rather than an optional preprocessing step. The implementation performs this centering as part of solving the problem rather than as a separate stabilization pass.

## Computational complexity

The algorithm has linear complexity in the number of input points. The dominant cost is evaluating the six independent sums in $M$, followed by eigendecomposing a fixed-size $3\times3$ symmetric matrix.

## Backend differences

- The NumPy backend performs all computations eagerly, using `numpy.linalg.eigh`.
- The PyTensor backend constructs the same computation symbolically, using `pytensor.tensor.linalg.eigh`, which supports automatic differentiation.
  The gradient formula assumes distinct eigenvalues; for point sets whose covariance matrix has (near-)repeated eigenvalues (e.g. an isotropic or highly symmetric point distribution), the eigenvector gradient becomes singular or numerically unstable.
  `test_flatness_gradient` verifies the gradient numerically for a well-separated (non-degenerate) point set.

## Current limitations

Both backends currently accept only 1-dimensional `x`, `y`, `z` — a single point set per call. The PyTensor backend validates this explicitly (`_validate_xyz_shapes`);
the NumPy backend does not yet perform this check.
Batched fitting (e.g. fitting many planes at once, as would arise from a plate/hierarchical structure in a PyMC model) is not yet supported.
Supporting it would require generalizing the covariance-matrix construction and the eigendecomposition to operate along a trailing axis while preserving leading batch dimensions, which PyTensor's `Blockwise` mechanism is designed to accommodate.
This is left as a future extension.

## References

- ISO 12781-1: Geometrical product specifications (GPS) — Flatness — Part 1: Vocabulary and parameters of flatness
- ISO 12781-2: Geometrical product specifications (GPS) — Flatness — Part 2: Specification operators
