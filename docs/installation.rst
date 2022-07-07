Installation
############
    
MicMec 1.0 has been developed with Python 3.7 and 3.8 in mind. 
It has been confirmed to work with the following combination of packages:

-   python      3.8.10
-   numpy       1.22.4
-   yaff        1.4.2
-   molmod      1.4.8
-   scipy       1.8.1
-   cython      0.29.30
-   matplotlib  3.5.2
-   h5py        3.7.0
-   jax         0.3.13 (optional)
-   jaxlib      0.3.10 (optional)
-   sphinx      5.0.2 (optional)

Sphinx is used for documentation, while JAX (with CUDA[CPU]) is used for automatic differentiation and just-in-time compilation.
Sphinx and JAX are entirely optional and do not affect the core routines of MicMec.

Yaff is an important dependency, even though it only supplies one module to MicMec: ``yaff.pes.ext.Cell``. That is, however, a crucial module. 
For now, we have not been able to decouple ``yaff.pes.ext.Cell`` from Yaff.
