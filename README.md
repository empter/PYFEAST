# PYFEAST
call [FEAST sparse eigensolver](http://www.ecs.umass.edu/~polizzi/feast/) in Python. Depends on python3, numpy, scipy, cython, and Intel MKL/OpenBLAS.

Solve standard HX=EX problem, H can be Real-Symmetric, Complex Hermitian, or General in [csr_matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) format.

# Setup
Step by step setup for **Ubuntu 18.08**, a recent linux dist. should also work.

1. download all files from **PYFEAST** in a folder, e.g., `pyfeast/`. **PYFEAST** alread contains FEAST source code and the directory stracture is slightly modified. If FEAST source code needs update, just replace `inlcude/` and `src/` under `FEAST/` folder by those corresponding folders in an offical release.

2. install essential packages.

* `sudo apt update` (optional)
  
* `sudo apt install build-essential gfortran`

3. (MKL user) install Intel MKL library, following [Intel web site](https://software.intel.com/content/www/us/en/develop/articles/installing-intel-free-libs-and-python-apt-repo.html), at lest install component **Intel® Math Kernel Library**.

* Activite MKL variables in terminal `source /opt/intel/mkl/bin/mklvars.sh intel64`

3. (openblas user) install openblas library, `sudo apt install libopenblas-dev`

4. install **Python3**, [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommanded.

* then install python packages `conda install numpy scipy cython`

* install **quspin** (optional) `conda install -c weinbe58 quspin`, if  quspin=0.3.4, then also `conda install numba=0.48.0`

5. build **FEAST** library.

* (MKL user) go to folder pyfeast/FEAST/src, run `make feast`

* (openblas user) go to folder pyfeast/FEAST/src, run `make MKL=no feast`

6. build **PYFEAST** python package.

* (MKL user) go to folder pyfeast/cython_feast, run `python3 setup.py build_ext install`

* (openblas user) go to folder pyfeast/cython_feast, run `python3 setup_openblas.py build_ext install`

7. see example, usage & test (**quspin** **JupyterNotebook/VSCode** needed, optional).

* check `quspin_feast_general.ipynb` for general matrix

* check `quspin_feast_hermitian.ipynb` for Hermitian system

# Uninstall or reinstall

* remove feast `pip3 uninstall feast` and reinstall **PYFEAST** python package (step 6).
