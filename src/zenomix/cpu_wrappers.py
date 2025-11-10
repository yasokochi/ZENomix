# Wrapper functions to force computation on CPU.
# These are useful when GPU/TPU memory is limited or when debugging numerical issues.

from .utils import _on_cpu
from .model import Model

def reconstruction_cpu(self):
    # Run the reconstruction method explicitly on CPU.
    # Ensures all intermediate JAX operations under this context use the CPU device.
    with _on_cpu():
        return Model.reconstruction(self)

def cov_gene_cpu(self, gene):
    # Compute covariance for a specific gene on CPU.
    with _on_cpu():
        return Model.cov_gene(self, gene)