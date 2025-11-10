# Gauss-Hermite quadrature for estimating expectations under the variational
# distribution q(X). The quadrature nodes (locs) and weights are defined in
# .constants and correspond to standardized Gaussian-Hermite quadrature rules.

import jax.numpy as jnp
from jax import jit, vmap
from .constants import locs, weights
from .kernel import kernel, kernel_diag

@jit
def get_psi(H, S, M, Xu, jitter, locs=locs, weights=weights):
    """
    Compute the psi statistics (ψ0, ψ1, ψ2) required by variational sparse GP
    methods using Gauss-Hermite quadrature.

    Args:
        H       : kernel hyperparameters
        S       : posterior variance of latent q(X) (shape: (n, q))
        M       : posterior mean of latent q(X) (shape: (n, q))
        Xu      : inducing points (shape: (m, q))
        jitter  : small jitter term added to kernel for numerical stability
        locs    : Gauss-Hermite quadrature nodes (precomputed)
        weights : Gauss-Hermite quadrature weights (precomputed)

    Returns:
        psi0    : (n,)     expectation of diagonal kernel term E_q[X][k(x,x)]
        psi1    : (n, m)   expectation of cross covariance E_q[X][k(x, Xu)]
        psi2    : (m, m)   expectation of Gram matrix E_q[X][Kfu^T Kfu]
    """
    # Convert scalar posterior variance to standard deviation
    S_sq = jnp.sqrt(S**2)

    # Prepare GH nodes and weights
    locs_j = jnp.asarray(locs)[:, None, None]      # (L, 1, 1)  GH quadrature nodes
    w_j    = jnp.asarray(weights)                  # (L,)      GH quadrature weights

    # Sample latent X using Gauss-Hermite: X = mu (M) + sigma (S_sq) * eps (locs_j)
    X = locs_j * S_sq + M[None, :, :]              # (L, n, q)

    # Evaluate kernel on each GH sample of X
    Kfu_L = vmap(lambda Xl: kernel(Xl, Xu, H, jitter))(X)   # (L, n, m)
    d_L   = vmap(lambda Xl: kernel_diag(Xl, H))(X)          # (L, n)

    # Quadrature-weighted expectations:
    psi0 = jnp.tensordot(w_j, d_L, axes=1)                 # (n,)
    psi1 = jnp.tensordot(w_j, Kfu_L, axes=1)               # (n, m)

    # Compute (Kfu)^T @ Kfu for each quadrature sample
    KTK_L = vmap(lambda K: K.T @ K)(Kfu_L)                 # (L, m, m)
    psi2  = jnp.tensordot(w_j, KTK_L, axes=1)              # (m, m)

    return psi0, psi1, psi2