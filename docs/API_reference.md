# ZENomix API Reference

## `ZENomix.Model`

```python
ZENomix.Model(
    data,
    reference,
    latent_dim=30,
    num_inducing=50,
    S2=1.0,
    sigma2=0.1,
    kernel_hyperparameters=None,
    jitter=None,
    prior_var=1.0,
    MMD_H=None,
    rff=True,
    rff_dim=2048,
    key=None,
)
```

Variational GPLVM–MMD model used in ZENomix.  
`data` (scRNA-seq) and `reference` (spatial reference) are jointly embedded into a shared latent
space, and the model can reconstruct gene expression on the reference domain
and estimate predictive uncertainty.

---

### **Parameters**

- **data** (*pandas.DataFrame*)  
  Target/query expression matrix (`cells × genes`).

- **reference** (*pandas.DataFrame*)  
  Reference expression matrix (`cells × genes`).  
  Only genes shared with `data` are used.

- **latent_dim** (*int*, default `30`)  
  Latent dimensionality.

- **num_inducing** (*int*, default `50`)  
  Number of inducing points.

- **S2** (*float*, default `1.0`)  
  Initial latent variance.

- **sigma2** (*float*, default `0.1`)  
  Initial observation noise variance.

- **kernel_hyperparameters** (*list[float]*, optional)  
  Initial GP kernel lengthscales. Defaults to `[0.01]`.

- **jitter** (*float*, optional)  
  Jitter for numerical stability.

- **prior_var** (*float*, default `1.0`)  
  Prior variance of latent variables.

- **MMD_H** (*list[float]*, optional)  
  Bandwidths of the MMD kernel. Defaults to `[0.01]`.

- **rff** (*bool*, default `True`)  
  Use Random Fourier Features for MMD.

- **rff_dim** (*int*, default `2048`)  
  Dimensionality of RFF embedding.

- **key** (*jax.random.PRNGKey*, optional)  
  Random key for initialization.

---

## `Model.latent_calibration`

```python
Model.latent_calibration(method="adam", **kwargs)
```

Optimize the latent variables and hyperparameters of the vGPLVM–MMD model.

### **Parameters**

- **method** (`"lbfgs"` or `"adam"`, default `"adam"`)  
  Optimization method.

- **kwargs** (depends on method):

  - When `method="lbfgs"`  
    - **scipy_options** (*dict*): Passed to `scipy.optimize.minimize`.

  - When `method="adam"`  
    - **steps** (*int*, default `500`)  
    - **lr** (*float*, default `1e-1`)  
    - **print_every** (*int*, default `10`)  
    - **log_every** (*int*, default `10`)  

---

## `Model.reconstruction`

```python
Y_hat = Model.reconstruction()
```

Reconstruct gene expression for the reference cells using the calibrated model.

### **Returns**

- **Y_hat** (*pandas.DataFrame*)  
  Reconstructed expression matrix of shape `(n_reference_cells, n_genes)`.  
  Columns correspond to genes; rows correspond to the reference index.

---

## `Model.cov_gene`

```python
var = Model.cov_gene(gene)
```

Estimate the predictive variance of reconstructed expression for a single gene.

### **Parameters**

- **gene** (*str*)  
  Gene name in `Model.data.columns`.

### **Returns**

- **var** (*jax.numpy.ndarray*)  
  1D array of length `n_reference_cells` containing predictive variances.

