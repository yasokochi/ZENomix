## src/zenomix/__init__.py

from .model import Model

from .constants import dtype, EPS, locs, weights
from .utils import _on_cpu, toVector, toParams, shapeParams, vv
from .kernel import kernel, kernel_diag, kernel_mmd
from .rff import rff_init_matern32, rff_features, MMD_rff
from .psi import get_psi
from .metrics import MMD
from .objective import (
    get_L_hat, get_KL, minus_L, minus_Lri,
    grad_minus_L, grad_minus_Lri,
)

# CPU wrappers
from .cpu_wrappers import reconstruction_cpu, cov_gene_cpu
Model.reconstruction_cpu = reconstruction_cpu
Model.cov_gene_cpu = cov_gene_cpu

__version__ = "0.9.0"

__all__ = [
    "Model",
    "dtype", "EPS", "locs", "weights",
    "_on_cpu", "toVector", "toParams", "shapeParams", "vv",
    "kernel", "kernel_diag", "kernel_mmd",
    "rff_init_matern32", "rff_features", "MMD_rff",
    "get_psi", "MMD",
    "get_L_hat", "get_KL", "minus_L", "minus_Lri",
    "grad_minus_L", "grad_minus_Lri",
]