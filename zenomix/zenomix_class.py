from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import grad, jit, vmap, lax
from jax import random
from sklearn.decomposition import PCA
from scipy import optimize, stats
import pandas as pd
import random

random.seed(0)


degree = 11
locs, weights = np.polynomial.hermite.hermgauss(degree)
locs *= np.sqrt(2.)
weights *= 1./np.sqrt(np.pi)


def toVector(M, S, sigma, kernel_hyperparameters):
    params = [S, sigma] + kernel_hyperparameters
    return jnp.hstack([M.flatten(), jnp.array(params)])


def toParams(vec, Xu, M, kernel_hyperparameters):
    m, q = Xu.shape
    n, q = M.shape
    num_hyperparameters = len(kernel_hyperparameters)
    return lax.dynamic_slice_in_dim(vec, 0, m*q).reshape(Xu.shape), lax.dynamic_slice_in_dim(vec, m*q, n*q).reshape(M.shape), lax.dynamic_slice_in_dim(vec, m*q + n*q, 1), lax.dynamic_slice_in_dim(vec, m*q + n*q + 1, 1), lax.dynamic_slice_in_dim(vec, m*q + n*q + 2, num_hyperparameters)


def shapeParams(Params, Xu, M_R, M_I, kernel_hyperparameters):
    m, q = Xu.shape
    nr, q = M_R.shape
    ni, q = M_I.shape
    num_hyperparameters = len(kernel_hyperparameters)
    Xu_p, r_p, i_p, h_p = lax.dynamic_slice_in_dim(Params, 0, m*q), lax.dynamic_slice_in_dim(Params, m*q, nr*q), lax.dynamic_slice_in_dim(Params, m*q + nr*q, ni*q), lax.dynamic_slice_in_dim(Params, m*q + nr*q + ni*q, num_hyperparameters+2)

    Params_r = jnp.hstack([Xu_p, r_p, h_p])
    Params_i = jnp.hstack([Xu_p, i_p, h_p])
    return Params_r, Params_i


vv = vmap(lambda x, y: jnp.outer(x, y))

###########
# Kernel
###########

@jit
def kernel(x, y, A, jitter=1e-10):
    YY = jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]**2
    VV = jnp.linalg.norm(y, axis=1)[jnp.newaxis, :]**2
    Dist_z = jnp.sqrt(YY + VV - 2*(x@y.T) + jitter)
    return (1 + jnp.sqrt(3)*(A[0]**2)*Dist_z)*jnp.exp(-jnp.sqrt(3)*(A[0]**2)*Dist_z)


@jit
def kernel_diag(x, A):
    return jnp.ones(x.shape[0])


@jit
def kernel_mmd(x, y, A, jitter=1e-10):
    YY = jnp.linalg.norm(x, axis=1)[:, jnp.newaxis]**2
    VV = jnp.linalg.norm(y, axis=1)[jnp.newaxis, :]**2
    Dist_z = jnp.sqrt(YY + VV - 2*(x@y.T) + jitter)
    return (1 + jnp.sqrt(3)*(A[0]**2)*Dist_z)*jnp.exp(-jnp.sqrt(3)*(A[0]**2)*Dist_z)


@jit
def MMD(M, Z, H, jitter):
    return kernel_mmd(M, M, H, jitter).mean() - 2*kernel_mmd(M, Z, H, jitter).mean() + kernel_mmd(Z, Z, H, jitter).mean()


@jit
def get_psi(H, S, M, Xu, jitter, locs=locs, weights=weights):
    n = M.shape[0]
    m = Xu.shape[0]
    S_sq = jnp.sqrt(S**2)
    psi0 = jnp.zeros((n,))
    psi1 = jnp.zeros((n, m))
    psi2 = jnp.zeros((m, m))
    for i in range(len(locs)):
        X = locs[i]*S_sq + M
        psi0 += weights[i]*kernel_diag(X, H)
        Kfu = kernel(X, Xu, H, jitter)
        psi1 += weights[i] * Kfu
        psi2 += weights[i] * (Kfu.T@Kfu)
    return psi0, psi1, psi2


@jit
def get_L_hat(A_i, Xu, M, S, sigma, H, jitter):
    n, p = A_i.shape

    Kuu = kernel(Xu, Xu, H, jitter)
    psi_0, psi_1, psi_2 = get_psi(H, S, M, Xu, jitter)
    s = jnp.sqrt(sigma**2)

    L = jnp.linalg.cholesky(Kuu)
    A = jsp.linalg.solve_triangular(L, psi_1.T, lower=True)/s
    tmp = jsp.linalg.solve_triangular(L, psi_2, lower=True)
    AAT = jsp.linalg.solve_triangular(L, tmp.T, lower=True)/(sigma**2)
    B = AAT + jnp.eye(AAT.shape[0])
    LB = jnp.linalg.cholesky(B)
    log_det_B = 2.0 * jnp.sum(jnp.log(jnp.diag(LB)))
    c = jsp.linalg.solve_triangular(LB, A@A_i, lower=True)/s

    bound = -0.5 * n*p * jnp.log(2 * jnp.pi * (sigma**2))
    bound += -0.5 * p * log_det_B
    bound += -0.5 * np.sum(A_i**2) / (sigma**2)
    bound += 0.5 * np.sum(c**2)
    bound += -0.5 * p * (np.sum(psi_0)/(sigma**2) - jnp.trace(AAT))

    return bound


@jit
def get_KL(M, S, prior_var):
    n, q = M.shape
    _S = (S**2)*jnp.ones((n, q))
    KL = -0.5*jnp.sum(jnp.log(_S))
    KL += 0.5*jnp.sum(jnp.log(prior_var*jnp.ones((n, q))))
    KL += 0
    KL -= 0.5 * n*q
    KL += 0.5*(jnp.sum(M**2) + jnp.sum(_S))/prior_var
    return KL


@jit
def minus_L(Params, A, Xu_shape, M_shape, kernel_hyperparameters, jitter, prior_var):
    Xu, M, S, sigma, H = toParams(
        Params, Xu_shape, M_shape, kernel_hyperparameters)
    return -(get_L_hat(A, Xu, M, S, sigma, H, jitter) - get_KL(M, S, prior_var))


@jit
def minus_Lri(Params, A_R, A_I, Xu_shape, M_R_shape, M_I_shape, init_val, kernel_hyperparameters, jitter, prior_var, MMD_H):
    Params_r, Params_i = shapeParams(
        Params, Xu_shape, M_R_shape, M_I_shape, kernel_hyperparameters)
    Xu = toParams(Params_r, Xu_shape, M_R_shape, kernel_hyperparameters)[0]
    M_R = toParams(Params_r, Xu_shape, M_R_shape, kernel_hyperparameters)[1]
    M_I = toParams(Params_i, Xu_shape,M_I_shape, kernel_hyperparameters)[1]
    return ((minus_L(Params_r, A_R, Xu_shape, M_R_shape, kernel_hyperparameters, jitter, prior_var) + minus_L(Params_i, A_I, Xu_shape, M_I_shape, kernel_hyperparameters, jitter, prior_var))/init_val[0] + MMD(M_R, M_I, MMD_H, jitter)/init_val[1]).reshape()


grad_minus_L = jit(grad(minus_L, argnums=0))

grad_minus_Lri = jit(grad(minus_Lri, argnums=0))


def wrapped_loss_vec_ri(A, arg):
    return Model.numpy_loss_vec_ri(arg, A)


def wrapped_grad_vec_ri(A, arg):
    return Model.numpy_grad_vec_ri(arg, A)


class Model():
    """
    vGPLVM-MMD model
    """

    def __init__(self, data, reference, latent_dim=30, kernel_hyperparameters=[0.01], num_inducing=50, S2=1., sigma2=0.1, jitter = 1e-10, prior_var=1., MMD_H = [0.01]):
        """
        initiation of GMM in Perler

        ---Paramerter---
        data: pandas.DataFrame object
        It contains scRNA-seq data (or other query data).
        Its each row represents each sample(cell) and each columns represents each genes.

        reference: pandas.DataFrame object
        It contains ISH data (or other reference data).
        Its each row represents each point(cell or region) and each columns represents each landmark genes.

        latent_dim: int
        The number of metagenes extracted by the partial least squares correlation analysis.
        The default is 30.
        
        kernel_hyperparameters: [float]
        the hyperparameters of the Gaussian process
        
        num_inducing: int
        the number of inducing points
        
        S2: float
        init values of variance of the posterior distributions
        
        sigma2: float
        init values of variacne of the Gaussian process

        ---Developers' parameter---
        Default values are recommended.

        MMD_H: [float]
        The method of Dimensionality Reduction in PERLER.
        'PLSC', 'PCA', and 'NA' (No DR) are implemented
        The default is 'PLSC'.
        """
        self.__data = data
        self.__reference = reference
        self.latent_dim = latent_dim
        self.__A_R = None
        self.__M_R = None
        self.__A_I = None
        self.__M_I = None
        self.__Xu = None
        self.__q = latent_dim
        self.__m = num_inducing
        self.__inducings = None
        self.__S = jnp.sqrt(S2)
        self.__kernel_hyperparameters_sq = [jnp.sqrt(k) for k in kernel_hyperparameters]
        self.__num_kernel_hyperparameters = len(kernel_hyperparameters)
        self.__jitter = jitter
        self.__prior_var = prior_var
        self.__MMD_H = [jnp.sqrt(h) for h in MMD_H]
        self.__sigma = jnp.sqrt(sigma2)
        self.init_Params = None
        self.logging_hyperparameters = None

        self.__initiate_params()

    def __initiate_params(self):
        """
        initiate parameter values of GP-Perler
        """
        A_R = self.__data[self.__reference.columns].values
        A_I = self.__reference.values
        self.__A_R = stats.zscore(A_R)
        self.__A_I = stats.zscore(A_I)
        self.__nr, self__p = self.__A_R.shape
        self.__ni, self__p = self.__A_I.shape
        
        # dim reduction (initiating GPLVM)
        pca = PCA(n_components=self.__q, svd_solver='full')
        pca.fit(self.__A_R)
        self.__M_I_init = pca.transform(self.__A_I)
        self.__M_I = self.__M_I_init.copy()
        self.__M_R_init = pca.transform(self.__A_R)
        self.__M_R = self.__M_R_init.copy()
        
        # randomly selecting inducing points
        RI = np.vstack([self.__M_R, self.__M_I])
        self.__inducings = random.sample([i for i in range(self.__nr + self.__ni)], self.__m)
        self.__Xu_init = RI[self.__inducings]
        self.__Xu = self.__Xu_init.copy()

        # initial parameters
        init_Params_y = self.__M_R.flatten()
        init_Params_v = toVector(self.__M_I, self.__S, self.__sigma, self.__kernel_hyperparameters_sq)
        self.num_params_y = len(init_Params_y)
        self.init_Params = jnp.hstack([self.__Xu.flatten(), init_Params_y, init_Params_v])

        # calculating cost values at initial parameters
        init_val_Ly = -(get_L_hat(self.__A_R, self.__Xu, self.__M_R, self.__S, self.__sigma, self.__kernel_hyperparameters_sq, self.__jitter) - get_KL(self.__M_R, self.__S, self.__prior_var))
        init_val_Lv = -(get_L_hat(self.__A_I, self.__Xu, self.__M_I, self.__S, self.__sigma, self.__kernel_hyperparameters_sq, self.__jitter) - get_KL(self.__M_I, self.__S, self.__prior_var))
        init_val_L = abs(init_val_Ly + init_val_Lv)
        init_val_MMD = abs(MMD(self.__M_R, self.__M_I, self.__MMD_H, self.__jitter))
        self._init_vals = [init_val_L, init_val_MMD]

    def numpy_loss_vec_ri(self, A):
        return np.array(minus_Lri(A, self.__A_R, self.__A_I, self.__Xu, self.__M_R, self.__M_I, self._init_vals, self.__kernel_hyperparameters_sq, self.__jitter, self.__prior_var, self.__MMD_H))

    def numpy_grad_vec_ri(self, A):
        return np.array(grad_minus_Lri(A, self.__A_R, self.__A_I, self.__Xu, self.__M_R, self.__M_I, self._init_vals, self.__kernel_hyperparameters_sq, self.__jitter, self.__prior_var, self.__MMD_H))

    def latent_calibration(self, scipy_options = None):
        
        '''
        matching the distributions of two data by mmd and gplvm
        '''

        self.logging_hyperparameters = [[self.__S**2, self.__sigma**2, self.__kernel_hyperparameters_sq[0]**2]]

        def callback(xk):
            self.logging_hyperparameters.append([i**2 for i in xk[-(self.__num_kernel_hyperparameters + 2):]])
            
        if scipy_options:
            options = scipy_options
        else:
            options = {'disp': None, 'maxcor': 10, 'ftol': 1e-6, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 10000, 'iprint': 1, 'maxls': 20, 'finite_diff_rel_step': None}
        self._res = optimize.minimize(fun=wrapped_loss_vec_ri, x0=self.init_Params, method="L-BFGS-B", jac=wrapped_grad_vec_ri, args=(self,), callback=callback, options=options)
        Params_r, Params_i = shapeParams(self._res.x, self.__Xu, self.__M_R, self.__M_I, self.__kernel_hyperparameters_sq)
        self.__Xu, self.__M_R, self.__S, self.__sigma, self.__kernel_hyperparameters_sq = toParams(Params_r, self.__Xu, self.__M_R, self.__kernel_hyperparameters_sq)
        self.__Xu, self.__M_I, self.__S, self.__sigma, self.__kernel_hyperparameters_sq = toParams(Params_i, self.__Xu, self.__M_I, self.__kernel_hyperparameters_sq)

    def reconstruction(self, geometry = None):
        rpsi0, rpsi1, rpsi2 = get_psi(self.__kernel_hyperparameters_sq, self.__S, self.__M_R, self.__Xu, self.__jitter)
        ipsi0, ipsi1, ipsi2 = get_psi(self.__kernel_hyperparameters_sq, self.__S, self.__M_I, self.__Xu, self.__jitter)
        beta = 1/self.__sigma**2
        Kuu = kernel(self.__Xu, self.__Xu, self.__kernel_hyperparameters_sq, self.__jitter)
        B = beta*jnp.linalg.inv(Kuu + beta*rpsi2)@rpsi1.T@self.__data
        self.__result = pd.DataFrame(np.array(ipsi1@B), columns = self.__data.columns)
        if geometry:
            self.__result_with_geometry =  pd.concat([geometry, self.__result], axis=1)

    def cov_gene(self, gene):
        rpsi0, rpsi1, rpsi2 = get_psi(self.__kernel_hyperparameters_sq, self.__S, self.__M_R, self.__Xu, self.__jitter)
        beta = 1/self.__sigma**2
        Kuu = kernel(self.__Xu, self.__Xu, self.__kernel_hyperparameters_sq, self.__jitter)
        Q_inv = jnp.linalg.inv(Kuu + beta*rpsi2)
        B = beta*Q_inv@rpsi1.T@self.__data[gene].values
        B = B.reshape(len(B), 1)

        def each_psi(z):
            z = z.reshape(1, len(z))
            ipsi0, ipsi1, ipsi2 = get_psi(self.__kernel_hyperparameters_sq, self.__S, z, self.__Xu, self.__jitter)
            cov = jnp.trace((ipsi2 - ipsi1.T@ipsi1)@vv(B.T, B.T), axis1=1, axis2=2) + ipsi0 - jnp.trace((jnp.linalg.inv(Kuu) - Q_inv)@ipsi2)
            return cov
        
        v_psi = vmap(each_psi)
        return v_psi(self.__M_I).reshape(self.__M_I.shape[0])

    # properties and setters (to access parameters)

    @property
    def data(self):
        return self.__data

    @property
    def A_R(self):
        return self.__A_R.copy()
    
    @property
    def A_I(self):
        return self.__A_I.copy()

    @property
    def M_R(self):
        return self.__M_R.copy()

    @property
    def M_I(self):
        return self.__M_I.copy()

    @property
    def Xu(self):
        return self.__Xu.copy()

    @property
    def M_R_init(self):
        return self.__M_R_init.copy()

    @property
    def M_I_init(self):
        return self.__M_I_init.copy()

    @property
    def Xu_init(self):
        return self.__Xu_init.copy()

    @property
    def S2(self):
        return self.__S**2

    @property
    def sigma2(self):
        return self.__sigma**2
    
    @property
    def sigma(self):
        return jnp.sqrt(self.__sigma**2)

    @property
    def kernel_hyperparameters(self):
        return [i**2 for i in self.__kernel_hyperparameters_sq]

    @property
    def result(self):
        return self.__result
    
    @property
    def result_with_geometry(self):
        return self.__result_with_geometry
