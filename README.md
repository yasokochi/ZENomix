# ZENomix: zero-shot reconstruction of mutant spatial transcriptomes

ZENomix is a novel computational method for reconstructing mutant spatial transcriptomes in azero-shot manner.
***
## Getting Started
### How to install
##### Dependencies and requirements
ZENomix is built upon the following packages...
- jax, numpy, pandas, scipy, scikit-learn, and matplotlib

If these modules do not exist, they are automatically installed.
##### installation
You can install ZENomix through [pip](https://pypi.org/project/pip/) command. Installation normally takes less than 1min. <br> 
pip
```python
pip install git+https://github.com/yasokochi/ZENomix.git
```
pipenv
```python
pipenv install git+https://github.com/yasokochi/ZENomix.git#egg=zenomix
```

### environment
The environment where ZENomix has been developed is below...

Mac OS
```python
!sw_vers
```

    ProductName:	Mac OS X
    ProductVersion:	10.15.7
    BuildVersion:	19H2


Python
```python
import sys
sys.version
```

    '3.10.5 (main, Jun 12 2022, 13:57:24) [Clang 12.0.0 (clang-1200.0.26.2)]'
also, the version information of the required modules is in [requirements.txt](requirements.txt)
***
## Usage
### ZENomix procedures
This is a very short introduction of ZENomix.<br>
As input, ZENomix take

- mutant scRNAseq data with sample rows and gene columns
- wildtype *in situ* data (ISH) with sample rows and gene columns


```python
import gzip
import pickle

import zenomix

#Initializing ZENomix
znx = zenomix.Model(data = scRNAseq, reference = ISH, latent_dim=20)
##The parameter fitting
znx.latent_calibration()
#spatial reconstruction
znx.reconstruction()
#save results
with gzip.open(dir_name, 'wb') as f:
    f.write(pickle.dumps(znx))
# The results can be seen as
pd.DataFrame(znx.result, columns = scRNAseq.columns)
```
