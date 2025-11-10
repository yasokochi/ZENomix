# Utility functions for ZENomix.

import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np
import pandas as pd
from scipy import sparse

from .constants import dtype

def _on_cpu():
    # Set the default device to CPU.
    return jax.default_device(jax.devices("cpu")[0])

def _ensure_common_genes(df_left: pd.DataFrame, df_right: pd.DataFrame):
    # Remove genes (columns) with zero variance to avoid numerical instability
    # threshold > 0 is used to prevent floating-point errors from treating ~0 variance as valid.
    threshold = 1e-12
    df_left = df_left.loc[:, df_left.var() > threshold]
    df_right = df_right.loc[:, df_right.var() > threshold]

    # Find overlapping gene names (intersection of columns)
    common = df_left.columns.intersection(df_right.columns)

    # If there are no common genes, further computations are meaningless
    if len(common) == 0:
        raise ValueError("There is no common gene between data and reference.")

    return common

def toVector(M, S, sigma, kernel_hyperparameters):
    # Flatten variational parameters into a single vector (used for optimization)
    # Vector = [flatten(M), S, sigma, kernel parameters...]
    params = [S, sigma] + kernel_hyperparameters
    return jnp.hstack([M.flatten(), jnp.array(params, dtype=dtype)])

def toParams(vec, Xu, M, kernel_hyperparameters):
    # Convert vectorized parameters back into structured components
    # Input vector layout:
    # [Xu (m*q), M (n*q), S, sigma, kernel_hyperparameters...]
    m, q = Xu.shape
    n, q = M.shape
    num_hyperparameters = len(kernel_hyperparameters)

    return (
        # inducing points Xu
        lax.dynamic_slice_in_dim(vec, 0, m*q).reshape(Xu.shape),
        # variational latent points M
        lax.dynamic_slice_in_dim(vec, m*q, n*q).reshape(M.shape),
        # S (scalar)
        lax.dynamic_slice_in_dim(vec, m*q + n*q, 1),
        # sigma (scalar)
        lax.dynamic_slice_in_dim(vec, m*q + n*q + 1, 1),
        # kernel hyperparameters
        lax.dynamic_slice_in_dim(vec, m*q + n*q + 2, num_hyperparameters),
    )

def shapeParams(Params, Xu, M_R, M_I, kernel_hyperparameters):
    # Similar to toParams(), but splits parameters into two sets:
    # one for (Xu, R) and one for (Xu, I). Useful when handling both
    # data and reference parameter groups separately.
    m, q = Xu.shape
    nr, q = M_R.shape
    ni, q = M_I.shape
    num_hyperparameters = len(kernel_hyperparameters)

    # Extract segments from Params vector
    Xu_p  = lax.dynamic_slice_in_dim(Params, 0, m*q)
    r_p   = lax.dynamic_slice_in_dim(Params, m*q, nr*q)
    i_p   = lax.dynamic_slice_in_dim(Params, m*q + nr*q, ni*q)
    h_p   = lax.dynamic_slice_in_dim(Params, m*q + nr*q + ni*q, num_hyperparameters+2)

    # Stack Xu + data and Xu + reference with same hyperparameters
    Params_r = jnp.hstack([Xu_p, r_p, h_p])
    Params_i = jnp.hstack([Xu_p, i_p, h_p])

    return Params_r, Params_i

# Vectorized outer product for batches.
# vv(x, y) computes outer(x[i], y[i]) for all batches using vmap.
vv = vmap(lambda x, y: jnp.outer(x, y))