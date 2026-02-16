# vGPLVM-MMD model implementation for ZENomix

from jax import config
config.update("jax_enable_x64", True)  # Enable 64-bit for numerical stability

from jax import random, jit, value_and_grad, vmap, device_get
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.cluster import kmeans_plusplus
from scipy import optimize, stats
import pandas as pd

from .constants import dtype, EPS
from .rff import rff_init_matern32, MMD_rff
from .utils import toVector, toParams, shapeParams, vv, _ensure_common_genes
from .objective import get_L_hat, get_KL, minus_Lri, grad_minus_Lri
from .reconstruction import gp_reconstruct_fp64, cov_gene_fp64
from .metrics import MMD
from .psi import get_psi
from .kernel import kernel

class Model():
    """
    vGPLVM-MMD model

    Args:
        data            : pandas.DataFrame
                          DataFrame of evaluation target data (cells x genes)
        reference       : pandas.DataFrame
                          DataFrame of reference data (cells x genes)
        latent_dim      : int
                          Dimensionality of latent space
        kernel_hyperparameters: list of float
                          Initial values of kernel hyperparameters (lengthscales)
        num_inducing   : int
                          Number of inducing points
        S2             : float
                          Initial value of latent variable variance
        sigma2         : float
                          Initial value of observation noise variance
        jitter         : float
                          Jitter for numerical stability in matrix computations
        prior_var      : float
                          Prior variance for latent variables
        MMD_H          : list of float
                          Initial values of MMD kernel bandwidths
        rff            : bool
                          Whether to use Random Fourier Features (RFF) for MMD computation
        rff_dim        : int
                          Dimensionality of RFF
        key            : jax.random.PRNGKey

    Methods:
        latent_calibration(method: str = 'adam', **kwargs):
            Latent space calibration of vGPLVM-MMD model using specified optimization method.
            Args:
                method: 'lbfgs' or 'adam'
                **kwargs:
                    - L-BFGS: scipy_options=dict(...)
                        For options, see scipy.optimize.minimize documentation for L-BFGS-B method.
                    - Adam: steps=int, lr=float, print_every=int, log_every=int

        reconstruction():
            Reconstruct gene expression data using the calibrated model.
            Returns:
                DataFrame of predicted geen expression data onto reference data

        cov_gene(gene):
            Compute the covariance of reconstructed gene expression for a specified gene.
            Args:
                gene: str
                      Gene name for which to compute covariance.
    """
    def __init__(
        self, data, reference, latent_dim=30, kernel_hyperparameters=None,
        num_inducing=50, S2=1., sigma2=0.1, jitter=None, prior_var=1.,
        MMD_H=None, rff: bool = True, rff_dim: int = 2048, key=None
    ):
        # Ensure both data and reference share common genes and drop zero-variance genes
        self.__genes_common = _ensure_common_genes(data, reference)

        # Keep references to (potentially large) dataframes; select common genes for reference
        self.__data = data
        self.__reference = reference.loc[:, self.__genes_common]

        # Core state (initialized later)
        self.__Xu = None
        self.__Xu_init = None
        self.__M_R_init = None
        self.__M_I_init = None
        self.__A_R = None
        self.__M_R = None
        self.__A_I = None
        self.__M_I = None
        self.__Xu = None

        # Model hyperparameters / shapes
        self.__q = latent_dim                  # latent dimension
        self.__m = num_inducing               # number of inducing points
        self.__inducings = None
        self.__S = jnp.sqrt(S2)               # latent std (S^2 given)
        if kernel_hyperparameters is None:
            kernel_hyperparameters = [0.01]
        # Internally store sqrt of hyperparameters (squared when accessed)
        self.__kernel_hyperparameters_sq = [jnp.sqrt(k) for k in kernel_hyperparameters]
        self.__num_kernel_hyperparameters = len(kernel_hyperparameters)
        if jitter is None:
            jitter = EPS
        self.__jitter = jitter
        self.__prior_var = prior_var
        if MMD_H is None:
            MMD_H = [0.01]
        # Store sqrt of MMD bandwidths
        self.__MMD_H = [jnp.sqrt(h) for h in MMD_H]
        self.__sigma = jnp.sqrt(sigma2)       # observation std (sigma^2 given)
        self.init_Params = None               # flattened parameter vector (optimizer state)
        self.logging_hyperparameters = None   # history of (S2, sigma2, kernel lengthscale^2)
        self.logging_mmd = None

        # Random key
        if key is None:
            key = random.PRNGKey(0)
        self.__key = key

        # RFF settings for MMD
        self.__use_rff = bool(rff)
        self.__rff_dim = rff_dim

        # Build initial latent variables and parameters
        self.__initiate_params()

    def __initiate_params(self):
        # Standardize count matrices
        A_R_raw = jnp.array(self.__data[self.__genes_common].values, dtype=dtype) 
        A_I_raw = jnp.array(self.__reference[self.__genes_common].values, dtype=dtype) 
        # Z-score normalization
        mean_r = jnp.mean(A_R_raw, axis=0)
        std_r = jnp.std(A_R_raw, axis=0)
        self.__A_R = (A_R_raw - mean_r) / std_r
        mean_i = jnp.mean(A_I_raw, axis=0) 
        std_i = jnp.std(A_I_raw, axis=0) 
        self.__A_I = (A_I_raw - mean_i) / std_i 
        
        self.__nr, self.__p = self.__A_R.shape
        self.__ni, _ = self.__A_I.shape 
        # PCA initialization
        U, S, Vt = jnp.linalg.svd(self.__A_R, full_matrices=False)
        components = Vt[:self.__q, :].T
        
        self.__M_R_init = jnp.dot(self.__A_R, components) 
        self.__M_I_init = jnp.dot(self.__A_I, components)

        self.__M_R = self.__M_R_init 
        self.__M_I = self.__M_I_init

        # Initialize inducing points with k-means++ on concatenated latents
        RI = np.vstack([self.__M_R, self.__M_I])
        centers, indices = kmeans_plusplus(RI, n_clusters=self.__m, random_state=42)
        self.__inducings = indices
        self.__Xu_init = jnp.array(centers, dtype=dtype)
        self.__Xu = self.__Xu_init.copy()

        # (Optional) Initialize RFF for MMD
        self.__key, subkey = random.split(self.__key)
        if self.__use_rff:
            self.__W, self.__b = rff_init_matern32(self.__q, self.__rff_dim, self.__MMD_H, subkey)
        else:
            self.__W, self.__b = None, None

        # Build flattened parameter vector: [Xu, M_R(flat), M_I + (S, sigma, kernel H)...]
        init_Params_y = self.__M_R.flatten()
        init_Params_v = toVector(self.__M_I, self.__S, self.__sigma, self.__kernel_hyperparameters_sq)
        self.num_params_y = len(init_Params_y)
        self.init_Params = jnp.hstack([self.__Xu.flatten(), init_Params_y, init_Params_v])

        # Initial objective values for scaling (|ELBO_data + ELBO_ref|, |MMD|)
        init_val_Ly = -(get_L_hat(self.__A_R, self.__Xu, self.__M_R, self.__S, self.__sigma, self.__kernel_hyperparameters_sq, self.__jitter) - get_KL(self.__M_R, self.__S, self.__prior_var))
        init_val_Lv = -(get_L_hat(self.__A_I, self.__Xu, self.__M_I, self.__S, self.__sigma, self.__kernel_hyperparameters_sq, self.__jitter) - get_KL(self.__M_I, self.__S, self.__prior_var))
        init_val_L = jnp.abs(init_val_Ly + init_val_Lv)
        
        init_val_MMD = jnp.abs(MMD_rff(self.__M_R, self.__M_I, self.__W, self.__b)) if (self.__W is not None and self.__b is not None) else jnp.abs(MMD(self.__M_R, self.__M_I, self.__MMD_H, self.__jitter))
        self._init_vals = [init_val_L, init_val_MMD]

        # Cast all arrays to the configured dtype for JAX computations
        self.__A_R = jnp.array(self.__A_R, dtype=dtype)
        self.__A_I = jnp.array(self.__A_I, dtype=dtype)
        self.__M_R = jnp.array(self.__M_R, dtype=dtype)
        self.__M_I = jnp.array(self.__M_I, dtype=dtype)
        self.__Xu = jnp.array(self.__Xu, dtype=dtype)
        self.init_Params = jnp.array(self.init_Params, dtype=dtype)
        self._init_vals = [jnp.array(self._init_vals[0], dtype=dtype), jnp.array(self._init_vals[1], dtype=dtype)]


    def _compute_normalized_mmd(self, params):
        params = jnp.asarray(params, dtype=dtype)
        Params_r, Params_i = shapeParams(params, self.__Xu, self.__M_R, self.__M_I, self.__kernel_hyperparameters_sq)
        M_R_cur = toParams(Params_r, self.__Xu, self.__M_R, self.__kernel_hyperparameters_sq)[1]
        M_I_cur = toParams(Params_i, self.__Xu, self.__M_I, self.__kernel_hyperparameters_sq)[1]
        mmd_val = (MMD_rff(M_R_cur, M_I_cur, self.__W, self.__b)
                   if (self.__W is not None and self.__b is not None)
                   else MMD(M_R_cur, M_I_cur, self.__MMD_H, self.__jitter))
        return float(mmd_val / self._init_vals[1])

    def numpy_loss_vec_ri(self, A):
        # Wrapper for SciPy: loss value as a NumPy array
        return np.array(minus_Lri(
            A, self.__A_R, self.__A_I,
            self.__Xu, self.__M_R, self.__M_I,
            self._init_vals,
            self.__kernel_hyperparameters_sq,
            self.__jitter,
            self.__prior_var,
            self.__MMD_H,
            self.__W,
            self.__b
        ))

    def numpy_grad_vec_ri(self, A):
        # Wrapper for SciPy: gradient as a NumPy array
        return np.array(grad_minus_Lri(
            A, self.__A_R, self.__A_I,
            self.__Xu, self.__M_R, self.__M_I,
            self._init_vals,
            self.__kernel_hyperparameters_sq,
            self.__jitter,
            self.__prior_var,
            self.__MMD_H,
            self.__W,
            self.__b
        ))
    
    def latent_calibration(self, method: str = 'adam', **kwargs):
        """
        latent space calibration of vGPLVM-MMD model of ZENomix. lbfgs (cpu only) or adam (supporting GPU calculation) optimizer is available.
        Args:
            method: 'lbfgs' or 'adam'
            **kwargs:
                - L-BFGS: scipy_options=dict(...)
                    For options, see scipy.optimize.minimize documentation for L-BFGS-B method.
                - Adam: steps=int, lr=float, print_every=int, log_every=int
        """
        m = method.lower()

        # L-BFGS-B (scipy.optimize) 
        if m in ('lbfgs', 'l-bfgs', 'l-bfgs-b'):
            # initialize logging with current (S^2, sigma^2, lengthscale^2)
            self.logging_hyperparameters = [[self.__S**2, self.__sigma**2, self.__kernel_hyperparameters_sq[0]**2]]
            self.logging_mmd = []

            # Callback logs hyperparameters along optimization path
            def callback(xk):
                self.logging_hyperparameters.append([i**2 for i in xk[-(self.__num_kernel_hyperparameters + 2):]])
                mmd_norm = self._compute_normalized_mmd(xk)
                self.logging_mmd.append(mmd_norm)
                print(f"  MMD = {mmd_norm:.6f}")

            # Default SciPy options if none provided
            options = kwargs.get('scipy_options', None)
            if not options:
                options = {'disp': False, 'maxcor': 10, 'ftol': 1e-6, 'gtol': 1e-05, 'eps': 1e-08,
                        'maxfun': 15000, 'maxiter': 10000, 'iprint': 1, 'maxls': 20, 'finite_diff_rel_step': None}

            # Wrap loss/grad to pass self via args (SciPy signature requirement)
            def wrapped_loss_vec_ri(A, arg):
                return Model.numpy_loss_vec_ri(arg, A)

            def wrapped_grad_vec_ri(A, arg):
                return Model.numpy_grad_vec_ri(arg, A)

            # Run L-BFGS-B
            self._res = optimize.minimize(
                fun=wrapped_loss_vec_ri,
                x0=self.init_Params,
                method="L-BFGS-B",
                jac=wrapped_grad_vec_ri,
                args=(self,),
                callback=callback,
                options=options
            )

            # Unpack optimized parameters for both modalities
            Params_r, Params_i = shapeParams(self._res.x, self.__Xu, self.__M_R, self.__M_I, self.__kernel_hyperparameters_sq)
            self.__Xu, self.__M_R, self.__S, self.__sigma, self.__kernel_hyperparameters_sq = toParams(
                Params_r, self.__Xu, self.__M_R, self.__kernel_hyperparameters_sq
            )
            self.__Xu, self.__M_I, self.__S, self.__sigma, self.__kernel_hyperparameters_sq = toParams(
                Params_i, self.__Xu, self.__M_I, self.__kernel_hyperparameters_sq
            )

        # Adam (JAX/Optax)
        elif m == 'adam':
            steps       = kwargs.get('steps', 500)
            lr          = kwargs.get('lr', 1e-1)
            print_every = kwargs.get('print_every', 10)
            log_every   = kwargs.get('log_every', 10)

            # initialize logging
            self.logging_hyperparameters = [[
                (self.__S**2).item(), (self.__sigma**2).item(), (self.__kernel_hyperparameters_sq[0]**2).item()
            ]]
            self.logging_mmd = []
            params = self.init_Params

            # Loss function for JAX optimization
            def loss_fn(p):
                return minus_Lri(
                    p,
                    self.__A_R, self.__A_I,
                    self.__Xu, self.__M_R, self.__M_I,
                    self._init_vals,
                    self.__kernel_hyperparameters_sq,
                    self.__jitter,
                    self.__prior_var,
                    self.__MMD_H,
                    self.__W,
                    self.__b
                )

            loss_and_grad = jit(value_and_grad(loss_fn))
            optim = optax.adam(lr)
            opt_state = optim.init(params)

            def step(p, state):
                # One optimizer step (compute loss, grads, and apply updates)
                loss, grads = loss_and_grad(p)
                updates, state = optim.update(grads, state, p)
                p = optax.apply_updates(p, updates)
                return p, state, loss

            # Main optimization loop
            for t in range(int(steps)):
                params, opt_state, loss = step(params, opt_state)
                should_print = (t % print_every) == 0 or (t == steps - 1)
                should_log   = (t % log_every) == 0 or (t == steps - 1)

                if should_print or should_log:
                    Params_r, Params_i = shapeParams(params, self.__Xu, self.__M_R, self.__M_I, self.__kernel_hyperparameters_sq)
                    _, _, S_cur, sigma_cur, H_cur = toParams(Params_r, self.__Xu, self.__M_R, self.__kernel_hyperparameters_sq)

                if should_print:
                    mmd_norm = self._compute_normalized_mmd(params)
                    print(f"Epoch {t}: loss = {loss.item()}, MMD = {mmd_norm:.6f}")

                if should_log:
                    self.logging_hyperparameters.append([
                        (S_cur**2).item(), (sigma_cur**2).item(), (H_cur[0]**2).item()
                    ])
                    if should_print:
                        self.logging_mmd.append(mmd_norm)
                    else:
                        self.logging_mmd.append(self._compute_normalized_mmd(params))

            # Store final parameters
            Params_r, Params_i = shapeParams(params, self.__Xu, self.__M_R, self.__M_I, self.__kernel_hyperparameters_sq)
            self.__Xu, self.__M_R, self.__S, self.__sigma, self.__kernel_hyperparameters_sq = toParams(
                Params_r, self.__Xu, self.__M_R, self.__kernel_hyperparameters_sq
            )
            self.__Xu, self.__M_I, self.__S, self.__sigma, self.__kernel_hyperparameters_sq = toParams(
                Params_i, self.__Xu, self.__M_I, self.__kernel_hyperparameters_sq
            )
            self.init_Params = params

        else:
            # Unknown optimizer specified
            raise ValueError(f"Unknown method '{method}'. Choose from 'lbfgs' or 'adam'.")
    
    def reconstruction(self):
        """
        Reconstruct gene expression data using the calibrated model.
        Returns:
            DataFrame of predicted geen expression data on reference data
        """

        Y_I = gp_reconstruct_fp64(
            self.__M_R,
            self.__M_I,
            self.__Xu,
            self.__S,
            self.__sigma,
            self.__kernel_hyperparameters_sq,
            self.__jitter,
            self.__data.values      # R 側データ
        )

        return pd.DataFrame(np.array(Y_I), columns=self.__data.columns, index=self.__reference.index)

    def cov_gene(self, gene):
        """
        Estimate variance of predicted gene expression data.
        Returns:
            variance vector of reconstructed gene expression for the specified gene.
        """
        cov_I = cov_gene_fp64(
            self.__M_R,
            self.__M_I,
            self.__Xu,
            self.__S,
            self.__sigma,
            self.__kernel_hyperparameters_sq,
            self.__jitter,
            self.__data[gene].values,
        )
        return cov_I.reshape(self.__M_I.shape[0])

    # properties
    @property
    def data(self): return self.__data
    @property
    def reference(self): return self.__reference
    @property
    def genes_common(self): return self.__genes_common
    @property
    def A_R(self): return self.__A_R.copy()
    @property
    def A_I(self): return self.__A_I.copy()
    @property
    def M_R(self): return self.__M_R.copy()
    @property
    def M_I(self): return self.__M_I.copy()
    @property
    def latent_dim(self): return self.__q
    @property
    def Xu(self):  return self.__Xu.copy()
    @property
    def M_R_init(self): return self.__M_R_init.copy()
    @property
    def M_I_init(self): return self.__M_I_init.copy()
    @property
    def Xu_init(self):  return self.__Xu_init.copy()
    @property
    def S2(self): return self.__S**2
    @property
    def sigma2(self): return self.__sigma**2
    @property
    def sigma(self): return jnp.sqrt(self.__sigma**2)
    @property
    def kernel_hyperparameters(self): return [i**2 for i in self.__kernel_hyperparameters_sq]
    @property
    def W(self): return self.__W
    @property
    def b(self): return self.__b
    @property
    def use_rff(self): return self.__use_rff
    @property
    def rff_dim(self): return self.__rff_dim