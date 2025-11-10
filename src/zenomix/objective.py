# Computing the objective function of the vGPLVM–MMD model of ZENomix

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad

from .kernel import kernel
from .psi import get_psi
from .metrics import MMD
from .rff import MMD_rff
from .constants import EPS, dtype
from .utils import toParams, shapeParams


# -----------------------------------------------------------------------------
# vGPLVM ELBO — likelihood term (log p(A | X))
# -----------------------------------------------------------------------------
def get_L_hat(A_i, Xu, M, S, sigma, H, jitter):
    # Convert parameters to double precision for ensuring numerical stability
    A_i = jnp.asarray(A_i, jnp.float64)
    Xu = jnp.asarray(Xu, jnp.float64)
    M = jnp.asarray(M, jnp.float64)
    S = jnp.asarray(S, jnp.float64)
    jitter = jnp.asarray(jitter, jnp.float64)

    n, p = A_i.shape  # n samples, p observed dimensions

    # Kernel over inducing points Kuu (m × m)
    Kuu = kernel(Xu, Xu, H, jitter)
    Kuu += EPS * jnp.eye(Kuu.shape[0])  # Numerical stabilization

    # Compute Phi-statistics required by variational GPLVM
    psi_0, psi_1, psi_2 = get_psi(H, S, M, Xu, jitter)

    # Variance of observation noise
    s = jnp.sqrt(sigma**2)

    # Compute Cholesky of Kuu
    L = jnp.linalg.cholesky(Kuu)

    # A = L^{-1} Phi_1^T / s
    A = jsp.linalg.solve_triangular(L, psi_1.T, lower=True) / s

    # AAT = (L^{-1} Phi_2 L^{-T}) / (sigma**2)
    tmp = jsp.linalg.solve_triangular(L, psi_2, lower=True)
    AAT = jsp.linalg.solve_triangular(L, tmp.T, lower=True) / (sigma**2)

    # B = AAT + I
    B = AAT + jnp.eye(AAT.shape[0])
    LB = jnp.linalg.cholesky(B)

    # log det(B)
    log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diag(LB)))

    # c = L_B^{-1} A A_i / σ
    c = jsp.linalg.solve_triangular(LB, A @ A_i, lower=True) / s

    # Variational lower bound on log marginal likelihood (ELBO)
    bound = -0.5 * n * p * jnp.log(2 * jnp.pi * (sigma**2))
    bound += -0.5 * p * log_det_B
    bound += -0.5 * jnp.sum(A_i**2) / (sigma**2)
    bound += 0.5 * jnp.sum(c**2)
    bound += -0.5 * p * (jnp.sum(psi_0) / (sigma**2) - jnp.trace(AAT))

    return bound.astype(dtype)


# -----------------------------------------------------------------------------
# vGPLVM ELBO — KL(q(X) || p(X)) term
# -----------------------------------------------------------------------------
def get_KL(M, S, prior_var):
    # KL divergence between q(X) from N(M, S^2) and p(X) ~ N(0, prior_var I)
    n, q = M.shape
    _S = (S**2) * jnp.ones((n, q))  # Expand variance if scalar

    KL = -0.5 * jnp.sum(jnp.log(_S))
    KL += 0.5 * jnp.sum(jnp.log(prior_var * jnp.ones((n, q))))
    KL -= 0.5 * n * q
    KL += 0.5 * (jnp.sum(M**2) + jnp.sum(_S)) / prior_var
    return KL


# -----------------------------------------------------------------------------
# Negative ELBO (for optimization)
# -----------------------------------------------------------------------------
def minus_L(Params, A, Xu_shape, M_shape, kernel_hyperparameters, jitter, prior_var):
    # Convert flattened parameter vector to structured tensors (Xu, M, S, sigma, H)
    Xu, M, S, sigma, H = toParams(Params, Xu_shape, M_shape, kernel_hyperparameters)

    # Return negative ELBO (optimizer minimizes this) for optimization
    return -(get_L_hat(A, Xu, M, S, sigma, H, jitter) - get_KL(M, S, prior_var))


# -----------------------------------------------------------------------------
# vGPLVM–MMD objective (multi-modal)
# ELBO (data) + ELBO (reference) + MMD regularization
# -----------------------------------------------------------------------------
def minus_Lri(Params, A_R, A_I, Xu_shape, M_R_shape, M_I_shape,
              init_val, kernel_hyperparameters, jitter, prior_var, MMD_H, W, b):
    # Split shared parameter vector into (data, reference)
    Params_r, Params_i = shapeParams(Params, Xu_shape, M_R_shape, M_I_shape, kernel_hyperparameters)

    # Extract individual parameters
    Xu  = toParams(Params_r, Xu_shape, M_R_shape, kernel_hyperparameters)[0]
    M_R = toParams(Params_r, Xu_shape, M_R_shape, kernel_hyperparameters)[1]
    M_I = toParams(Params_i, Xu_shape, M_I_shape, kernel_hyperparameters)[1]

    # Compute ELBO for each modality
    L_r = minus_L(Params_r, A_R, Xu_shape, M_R_shape, kernel_hyperparameters, jitter, prior_var)
    L_i = minus_L(Params_i, A_I, Xu_shape, M_I_shape, kernel_hyperparameters, jitter, prior_var)

    # MMD regularization to align latent distributions M_R and M_I
    mmd_value = (MMD_rff(M_R, M_I, W, b)
                 if (W is not None and b is not None)
                 else MMD(M_R, M_I, MMD_H, jitter))

    # Return weighted final loss (scaled for stability)
    return (((L_r + L_i) / init_val[0]) + (mmd_value / init_val[1])).reshape()


# -----------------------------------------------------------------------------
# Gradients with respect to parameters (for optimization by scipy.optimize)
# -----------------------------------------------------------------------------
grad_minus_L = jit(grad(minus_L, argnums=0))
grad_minus_Lri = jit(grad(minus_Lri, argnums=0))