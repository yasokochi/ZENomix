# Constants used throughout ZENomix

import jax.numpy as jnp
import numpy as np

# Default JAX floating point dtype used in the model
dtype = jnp.float32

# Small epsilon added for numerical stability (e.g., in Cholesky and kernel computations)
EPS = 1e-10

# Gauss–Hermite quadrature constants
# the number of quadrature nodes (higher = more accurate, slower)
degree = 11

# Compute Gauss–Hermite quadrature nodes (locs) and weights
locs, weights = np.polynomial.hermite.hermgauss(degree)

# Scale nodes and weights so that quadrature approximates expectation 
# under the Standard normal distribution N(0, 1) instead of exp(-x^2)
locs *= np.sqrt(2.)          # Adjust nodes (xi * sqrt(2))
weights *= 1. / np.sqrt(np.pi)   # Adjust weights so integration approximates E[f(X)]
# Convert numpy arrays to jax.numpy arrays with the correct dtype
locs = jnp.array(locs, dtype=dtype)
weights = jnp.array(weights, dtype=dtype)