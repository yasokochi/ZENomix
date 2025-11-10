# Random Fourier feature (RFF) for accelerating MMD computation.
# This implementation references the GPJax Matern 3/2 RFF approach.
# citation: GPJax (https://doi.org/10.21105/joss.04455)

import jax.numpy as jnp
from jax import jit, random
# numpyro is used to sample from the Student-t distribution (needed for Matern 3/2 RFF).
try:
    import numpyro.distributions as npd
except Exception:
    npd = None

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

def rff_init_matern32(d, D, A, key):
    '''
    Initialize Random Fourier Feature parameters (W and b) for approximating
    the Matern 3/2 kernel using RFF.

    Args:
        d:      latent dimension (after PCA projection)
        D:      number of random Fourier features (RFF dimensionality)
        A:      kernel bandwidth parameter (list or array-like)
        key:    jax.PRNGKey used for sampling

    Returns:
        W:      (D, d) jnp array (RFF weights, sampled from Student-t)
        b:      (D,)   jnp array (RFF phase offsets, uniform on [0, 2π))
    '''
    ell = 1.0 / (A[0] ** 2)        # Convert bandwidth to lengthscale
    df = 3.0                       # Matern 3/2 kernel → Student-t df = 3
    scale = jnp.sqrt(df) / ell     # Scale parameter of Student-t distribution

    # Split PRNG key for independent sampling of W and b
    key_w, key_b = random.split(key, 2)

    # Student-t sampling requires numpyro
    if npd is None:
        raise ImportError('numpyro is required for StudentT sampling in rff_init_matern32. Please install numpyro or set rff=False.')

    # Sample RFF weight matrix W from StudentT(df, loc=0, scale)
    W = npd.StudentT(df=df, loc=0.0, scale=scale).sample(key_w, (D, d))

    # Sample phase offset b ~ Uniform(0, 2π)
    b = random.uniform(key_b, (D,)) * (2.0 * jnp.pi)

    return W, b

@jit
def rff_features(X, W, b):
    # Compute cosine RFF embedding: Phi(x) = sqrt(2/D) * cos(Wx + b)
    WX = X @ W.T
    return jnp.cos(WX + b) * jnp.sqrt(2.0 / W.shape[0])

@jit
def MMD_rff(X, Y, W, b):
    # Approximate MMD using RFF embedding: MMD ≈ ||muX - muY||^2
    ZX = rff_features(X, W, b)
    ZY = rff_features(Y, W, b)
    muX = ZX.mean(axis=0)
    muY = ZY.mean(axis=0)
    return jnp.sum((muX - muY) ** 2)


def lengthscale_heuristic(
    data_df,
    reference_df,
    A_min=1e-3,
    A_max=10.0,
    num_A=40,
    latent_dim=50,
    D=2048,
    dtype=jnp.float32,
    key=random.PRNGKey(0),
):
    """
    Heuristic for selecting the RFF MMD kernel bandwidth (lengthscale A).

    Procedure:
      1. Normalize and apply PCA to project data into latent_dim space.
      2. Perform grid search over A in log space.
      3. For each A: generate RFF (W, b) and compute approximate MMD.
      4. Return the A that maximizes MMD.

    Args:
        data_df      : pandas DataFrame of scRNA-seq data
        reference_df : pandas DataFrame of spatial reference data
        A_min, A_max : search range for kernel bandwidth (log-scaled)
        num_A        : number of grid points to evaluate between A_min and A_max
        latent_dim   : dimensionality of latent space
        D            : number of random Fourier features (RFF dimensionality)
        dtype        : JAX dtype
        key          : jax.random.PRNGKey for reproducible sampling

    Returns:
        best_A       : kernel bandwidth parameter A that maximizes MMD
        best_mmd     : corresponding MMD value that maximizes MMD
        A_grid       : list of evaluated kernel bandwidth parameters A values
        mmd_values   : MMD values computed for each kernel bandwidth parameter A in the grid
    """

    # Normalize (z-score) the shared genes for PCA projection
    common_cols = reference_df.columns
    A_R = stats.zscore(data_df[common_cols].values)
    A_I = stats.zscore(reference_df.values)

    # Fit PCA on data and transform both datasets into latent space
    pca = PCA(n_components=latent_dim, svd_solver="arpack")
    pca.fit(A_R)
    M_R = jnp.array(pca.transform(A_R), dtype=dtype)
    M_I = jnp.array(pca.transform(A_I), dtype=dtype)

    # Log-scale grid search for kernel lengthscale A
    A_grid = jnp.logspace(jnp.log10(A_min), jnp.log10(A_max), num_A)
    mmd_values = []

    d = latent_dim

    # Evaluate MMD via RFF for each A
    for i, A in enumerate(A_grid):
        key, subkey = random.split(key)
        W, b = rff_init_matern32(d, D, [jnp.sqrt(A)], subkey)
        mmd = MMD_rff(M_R, M_I, W, b)
        mmd_values.append(mmd)

    mmd_values = jnp.array(mmd_values)

    # Pick the A that maximizes MMD
    idx = int(jnp.argmax(mmd_values))
    best_A = float(A_grid[idx])
    best_mmd = float(mmd_values[idx])

    print(f"[best] A = {best_A:.6f},  MMD_rff = {best_mmd:.6f}")

    return [best_A], best_mmd, A_grid, mmd_values