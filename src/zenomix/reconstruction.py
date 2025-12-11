# Computing the objective function of the vGPLVM–MMD model of ZENomix

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, grad

from .kernel import kernel
from .psi import get_psi
from .metrics import MMD
from .rff import MMD_rff
from .constants import EPS, dtype
from .utils import toParams, shapeParams, vmap


def gp_reconstruct_fp64(
    M_R, M_I, Xu, S, sigma, kernel_hyperparams, jitter, data_matrix
):
    """
    Inputs:
        M_R, M_I : latent means (R and I)
        Xu      : inducing points
        S       : posterior variance (scalar)
        sigma   : GP noise standard deviation
        kernel_hyperparams : kernel hyperparameters (H)
        jitter  : jitter
        data_matrix : R-side observed data (n_R x p)

    Output:
        Y_I : reconstruction (n_I x p)
    """

    # ★ Cast everything to FP64
    M_R = M_R.astype(jnp.float64)
    M_I = M_I.astype(jnp.float64)
    Xu = Xu.astype(jnp.float64)
    S = jnp.asarray(S, jnp.float64)
    sigma = jnp.asarray(sigma, jnp.float64)
    H = jnp.asarray(kernel_hyperparams, jnp.float64)
    jitter = jnp.asarray(jitter, jnp.float64)
    data_matrix = jnp.asarray(data_matrix, dtype=dtype)

    # ------ psi computation (fp64) ------
    rpsi0, rpsi1, rpsi2 = get_psi(H, S, M_R, Xu, jitter)
    ipsi0, ipsi1, ipsi2 = get_psi(H, S, M_I, Xu, jitter)

    # ------ Kuu + EPSI ------
    Kuu = kernel(Xu, Xu, H, jitter) + EPS * jnp.eye(Xu.shape[0], dtype=jnp.float64)

    # ------ Same computation as get_L_hat ------
    s = jnp.sqrt(sigma**2)
    L = jnp.linalg.cholesky(Kuu)

    A = jsp.linalg.solve_triangular(L, rpsi1.T, lower=True) / s
    tmp = jsp.linalg.solve_triangular(L, rpsi2, lower=True)
    AAT = jsp.linalg.solve_triangular(L, tmp.T, lower=True) / (sigma**2)
    B_mat = AAT + jnp.eye(AAT.shape[0], dtype=jnp.float64)

    # ------ B^{-1} A ------
    LB = jnp.linalg.cholesky(B_mat)
    Y_mid = jsp.linalg.solve_triangular(LB, A, lower=True)
    BinvA = jsp.linalg.solve_triangular(LB.T, Y_mid, lower=False)

    # ------ C = L^{-T} B^{-1} A ------
    C = jsp.linalg.solve_triangular(L.T, BinvA, lower=False).astype(dtype=dtype)

    # ------ β * Qinv @ rpsi1.T @ data = (1/sigma) * C @ data ------
    B_coef = ((1.0 / s) * (C @ data_matrix)).astype(dtype=dtype)

    # ------ Reconstruction on the I side ------
    Y_I = (ipsi1 @ B_coef).astype(dtype=dtype)

    return Y_I


def cov_gene_fp64(
    M_R, M_I, Xu, S, sigma, kernel_hyperparams, jitter,
    data_gene_vector,
):
    """
    Inputs:
        M_R : (n_R, q)  latent mean on the R side
        M_I : (n_I, q)  latent mean on the I side
        Xu  : (m, q)    inducing points
        S   : scalar (shared posterior std or its sqrt)
        sigma : scalar (GP noise std)
        kernel_hyperparams : kernel hyperparameters H
        jitter : jitter
        data_gene_vector : (n_R,)  observed vector for this gene (self.__data[gene].values)

    Output:
        cov_I : (n_I,)  covariance at each I-side point
    """

    # ---------- Cast everything to float64 ----------
    M_R = jnp.asarray(M_R, jnp.float64)
    M_I = jnp.asarray(M_I, jnp.float64)
    Xu  = jnp.asarray(Xu,  jnp.float64)
    S   = jnp.asarray(S,   jnp.float64)
    sigma = jnp.asarray(sigma, jnp.float64)
    H   = jnp.asarray(kernel_hyperparams, jnp.float64)
    jitter = jnp.asarray(jitter, jnp.float64)
    y   = jnp.asarray(data_gene_vector, jnp.float64)   # (n_R,)

    # ---------- Compute psi on the R side ----------
    rpsi0, rpsi1, rpsi2 = get_psi(H, S, M_R, Xu, jitter)
    # rpsi1: (n_R, m), rpsi2: (m, m)

    beta = 1.0 / (sigma**2)

    # ---------- Kuu and its Cholesky ----------
    Kuu = kernel(Xu, Xu, H, jitter)
    Kuu = Kuu + EPS * jnp.eye(Kuu.shape[0], dtype=jnp.float64)

    L = jnp.linalg.cholesky(Kuu)  # Kuu = L L^T

    # ---------- Compute inv(Kuu) from Cholesky ----------
    m = Kuu.shape[0]
    I_m = jnp.eye(m, dtype=jnp.float64)
    # invKuu = L^{-T} L^{-1}
    invKuu = jsp.linalg.cho_solve((L, True), I_m)

    # ---------- Same A, AAT, B computation as get_L_hat ----------
    s = jnp.sqrt(sigma**2)

    # A = L^{-1} rpsi1.T / s
    A = jsp.linalg.solve_triangular(L, rpsi1.T, lower=True) / s      # (m, n_R)

    # tmp = L^{-1} rpsi2
    tmp = jsp.linalg.solve_triangular(L, rpsi2, lower=True)          # (m, m)
    # AAT = L^{-1} rpsi2 L^{-T} / sigma^2
    AAT = jsp.linalg.solve_triangular(L, tmp.T, lower=True) / (sigma**2)  # (m, m)

    B_mat = AAT + jnp.eye(m, dtype=jnp.float64)                      # (m, m)

    # ---------- B^{-1} (Cholesky) ----------
    LB = jnp.linalg.cholesky(B_mat)
    Y_mid = jsp.linalg.solve_triangular(LB, I_m, lower=True)
    Binv = jsp.linalg.solve_triangular(LB.T, Y_mid, lower=False)     # B^{-1}

    # ---------- Q_inv = L^{-T} B^{-1} L^{-1} ----------
    # L^{-1}
    Linv = jsp.linalg.solve_triangular(L, I_m, lower=True)           # (m, m)
    Q_inv = Linv.T @ Binv @ Linv                                     # (m, m)

    # ---------- B vector: beta * Q_inv @ rpsi1.T @ y ----------
    By = rpsi1.T @ y                      # (m,)
    B_vec = beta * (Q_inv @ By)           # (m,)

    # Rank-1 outer product B B^T
    BBt = jnp.outer(B_vec, B_vec)         # (m, m)

    # ---------- Compute covariance for each I-side point z ----------
    def each_cov(z):
        z = z.reshape(1, -1)  # (1, q)
        ipsi0, ipsi1, ipsi2 = get_psi(H, S, z, Xu, jitter)
        # ipsi1: (1, m), ipsi2: (m, m), ipsi0: (1,)

        # (ipsi2 - ipsi1^T ipsi1) and trace with B B^T
        ipsi1T = ipsi1.T                                  # (m, 1)
        diff = ipsi2 - ipsi1T @ ipsi1                    # (m, m)
        term1 = jnp.trace(diff @ BBt)                    # scalar

        # ipsi0 is (1,), extract the element
        term_psi0 = ipsi0[0]

        # tr((invKuu - Q_inv) @ ipsi2)
        term2 = jnp.trace((invKuu - Q_inv) @ ipsi2)

        cov = term1 + term_psi0 - term2
        return cov

    v_each = vmap(each_cov)
    cov_I = v_each(M_I)   # (n_I,)

    return cov_I.astype(jnp.float64)