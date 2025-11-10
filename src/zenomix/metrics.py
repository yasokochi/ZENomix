# Full Maximum Mean Discrepancy (MMD) computation using the Matern 3/2 kernel

from jax import jit
from .kernel import kernel_mmd

@jit
def MMD(M, Z, H, jitter):
    """
    Compute squared Maximum Mean Discrepancy (MMD^2) between two latent distributions.

        MMD^2 = E[k(M, M)] - 2 E[k(M, Z)] + E[k(Z, Z)]

    where k(·,·) is the kernel function (Matern 3/2 here).

    Args:
        M : latent embedding for scRNA-seq data (shape: (n, q))
        Z : latent embedding for spatial reference data (shape: (m, q))
        H : kernel hyperparameters (lengthscale)
        jitter : small epsilon added for numerical stability

    Returns:
        Scalar MMD^2 value (non-negative)
    """

    # Mean of kernel over pairs within M
    term_MM = kernel_mmd(M, M, H, jitter).mean()

    # Cross-kernel expectation between M and Z
    term_MZ = kernel_mmd(M, Z, H, jitter).mean()

    # Mean of kernel over pairs within Z
    term_ZZ = kernel_mmd(Z, Z, H, jitter).mean()

    # MMD formula
    return term_MM - 2 * term_MZ + term_ZZ + jitter