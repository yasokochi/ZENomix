# Basic kernel functions used in the model (Matern 3/2 kernel)

import jax.numpy as jnp
from jax import jit



# Matern 3/2 kernel (for GP and inducing-point computations)
@jit
def kernel(x, y, A, jitter=1e-10):
    """
    Matern 3/2 kernel with lengthscale parameterized via A.
    k(x, y) = (1 + sqrt(3) * l * dist(x, y)) * exp(-sqrt(3) * l * dist(x, y))
      where l = A[0]^2

    Args:
        x:      (n, q) input matrix
        y:      (m, q) input matrix
        A:      kernel hyperparameters (expects A[0] as lengthscale parameter)
        jitter: small constant added for numerical stability

    Returns:
        (n, m) kernel matrix
    """

    # Squared norm terms ||x||^2 and ||y||^2 for pairwise distance computation
    YY = jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]**2    # (n, 1)
    VV = jnp.linalg.norm(y, axis=1)[jnp.newaxis, :]**2    # (1, m)

    # Pairwise squared distance
    sq = YY + VV - 2 * (x @ y.T)
    sq = jnp.maximum(sq, jitter)  # Numerical stability (avoid negative values)
    Dist_z = jnp.sqrt(sq)

    # Matern 3/2 closed-form kernel
    return (1 + jnp.sqrt(3) * (A[0]**2) * Dist_z) * jnp.exp(-jnp.sqrt(3) * (A[0]**2) * Dist_z)


# Diagonal of the Matern 3/2 kernel
@jit
def kernel_diag(x, A):
    """
    Diagonal of the kernel matrix k(x, x).
    For Matern kernels, k(x_i, x_i) = 1 (scaled).
    """
    return jnp.ones(x.shape[0], dtype=x.dtype)

# Matern 3/2 kernel used specifically for MMD computation (same as kernel)
@jit
def kernel_mmd(x, y, A, jitter=1e-10):
    """
    Matern 3/2 kernel used for MMD-based latent space alignment.
    Identical formula to 'kernel()', separated for clarity.
    """

    YY = jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]**2
    VV = jnp.linalg.norm(y, axis=1)[jnp.newaxis, :]**2

    sq = YY + VV - 2 * (x @ y.T)
    sq = jnp.maximum(sq, jitter)
    Dist_z = jnp.sqrt(sq)

    return (1 + jnp.sqrt(3) * (A[0]**2) * Dist_z) * jnp.exp(-jnp.sqrt(3) * (A[0]**2) * Dist_z)