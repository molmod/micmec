Installation
############
    
Dependencies
============

MicMec 1.0 has been developed with Python 3.7 and 3.8 in mind. 
It has been confirmed to work with the following combination of packages:

-   python      3.8.10
-   numpy       1.22.4
-   molmod      1.4.8
-   scipy       1.8.1
-   cython      0.29.30
-   matplotlib  3.5.2
-   h5py        3.7.0
-   jax         0.3.13 (optional)
-   jaxlib      0.3.10 (optional)
-   sphinx      5.0.2 (optional)
-   setuptools  63.0.1 (optional)

Sphinx is used for documentation, while JAX (with CUDA[CPU]) is used for automatic differentiation and just-in-time compilation.
Sphinx and JAX are entirely optional and do not affect the core routines of MicMec. SetupTools is also optional.


Installation
============

MicMec can be installed manually by cloning the entire GitHub repository into a pre-existing directory on your Python path, for instance, ``micmec``.

.. code:: bash

   git clone https://github.com/Jlvdwall/micmec.git micmec
   python

>>> import sys 
>>> repo = "micmec"
>>> any((repo == directory[-len(repo):]) for directory in sys.path)

The Python code checks whether the repository directory (``micmec`` in this case) is on your Python path. If it is not, you must add it to the path by editing your ``.bashrc`` or ``.profile`` script. With that, you can use MicMec to its fullest extent. As a bonus, you can install MicMec locally, by performing the following commands.

.. code:: bash

   cd micmec
   pip install .
   pip list

That will add MicMec to the list of installed packages.




