# ZENOmix

ZENOmix enables zero-shot reconstruction of mutant spatial transcriptomes using scRNA-seq and wildtype *in situ* data.

---

## 1. Install ZENOmix

You can install ZENOmix from Github directly via `pip`:

```bash
pip install git+https://github.com/yasokochi/ZENOmix.git
```

Or clone and install in editable/development mode:

```bash
git clone https://github.com/yasokochi/ZENOmix.git
cd ZENOmix
pip install -e .
```

> ❗ **Important:** ZENOmix is built on **JAX**.  
> **Recommended:** Install JAX yourself first (CPU or GPU) following the official guidance for your environment.  
> **Alternative:** If you haven’t installed JAX yet, you can also install it **together** using extras such as `zenomix[cuda12]`.


---

## 2. Install JAX (recommended to do first)

Install JAX **before** ZENOmix depending on your hardware/runtime.  
*MPS/Metal (Mac GPU) is not supported due to float64 precision requirements.*

### 🖥️ CPU only

```bash
pip install --upgrade "jax[cpu]>=0.6,<0.7"
```

### ⚡ GPU (CUDA 12.x)

```bash
pip install --upgrade "jax[cuda12]>=0.6,<0.7"
```

### ⚡ GPU (CUDA 13.x)

```bash
pip install --upgrade "jax[cuda13]>=0.6,<0.7"
```

> After JAX is installed, install ZENOmix normally (Section 1).


---

## 🔧 Didn’t install JAX yet? (alternative via extras)

You can install ZENOmix **and JAX** together using extras:

| Environment | Command |
|-------------|---------|
| CPU only    | `pip install "zenomix[cpu] @ git+https://github.com/yasokochi/ZENOmix.git"` |
| CUDA 12.x   | `pip install "zenomix[cuda12] @ git+https://github.com/yasokochi/ZENOmix.git"` |
| CUDA 13.x   | `pip install "zenomix[cuda13] @ git+https://github.com/yasokochi/ZENOmix.git"` |


## 🔧 Optional: RFF acceleration (Random Fourier Features)
```bash
pip install "zenomix[rff] @ git+https://github.com/yasokochi/ZENOmix.git"
```

RFF enables fast MMD evaluation using `numpyro` (Student‑t sampling).

### ➕ Combine options

```bash
pip install "zenomix[cuda12,rff] @ git+https://github.com/yasokochi/ZENOmix.git"
```


---

## 3. Verify GPU is detected

```python
import jax
import jax.numpy as jnp

print(jax.default_backend())  # expect 'gpu' or 'cuda'
print(jax.devices())          # should list GPU
```


---

## 4. Basic Usage Example

```python
import pandas as pd
import zenomix

# If your data are AnnData objects (scanpy), pass adata.to_df() (cells × genes).
# Make sure both datasets are normalized (e.g., log1p(CPM)).
znx = zenomix.Model(
    data=scRNAseq,   # mutant scRNA-seq (cells × genes) DataFrame or compatible
    reference=ISH,   # WT ISH spatial reference (cells × genes) DataFrame or compatible
    latent_dim=20,
)

znx.latent_calibration(method='adam')
recon = znx.reconstruction()
print(recon.head())
```


---

## 5. Example environments (Mac / Linux)

### ✅ macOS (CPU only)

```bash
# Mac Studio (2025, Apple M3 Ultra, 256GB Unified Memory)
# macOS: 15.5 Sequoia
# Python: 3.11.5

# without Jax
pip install git+https://github.com/yasokochi/ZENOmix.git
# with Jax
pip install "zenomix[cpu] @ git+https://github.com/yasokochi/ZENOmix.git"
```

### ✅ Linux + CUDA

```bash
# AMD Ryzen Threadripper PRO 3955WX (16 Cores)
# 4x DDR4-3200 32 GB RAM (128GB in total)
# 1x NVIDIA RTX 6000Ada
# OS: Ubuntu 20.04
# NVIDIA Driver: 535.230.02
# CUDA: 12.2
# Python: 3.11.5

# without Jax
pip install git+https://github.com/yasokochi/ZENOmix.git
# PyPI with Jax and Numpyro
pip install "zenomix[cuda12,rff] @ git+https://github.com/yasokochi/ZENOmix.git"
```


---

## 6. Runtime / memory reference

Dataset: 50k scRNA-seq cells, 5k reference cells.

| Mode | Time | Memory |
|------|------|--------|
| RFF + GPU | ~3 min | ~3 GB VRAM |
| Full MMD + GPU | ~5 min | ~10 GB VRAM |
| Full MMD + CPU | ~15 min | ~10 GB RAM |


---

## 7. Summary

| Scenario         | Command                              |
|------------------|--------------------------------------|
| Recommended      | Install JAX first, then ZENomix      |
| CPU only         | `pip install "zenomix[cpu] @ git+https://github.com/yasokochi/ZENOmix.git"`         |
| GPU (CUDA 12.x)  | `pip install "zenomix[cuda12] @ git+https://github.com/yasokochi/ZENOmix.git"`      |
| GPU + RFF        | `pip install "zenomix[cuda12,rff] @ git+https://github.com/yasokochi/ZENOmix.git"`  |

