Creating micromechanical cell types
###################################

Trajectory analysis
===================

The micromechanical model is based on the concept of nanocell types. The elastic properties of nanocell types in a micromechanical system determine the interactions between the micromechanical nodes. These coarse-grained parameters can be extracted from simulations on a higher level of theory.

In general, we can use an HDF5 trajectory file from an atomistic MD simulation to obtain everything we need to know about a nanocell type. Firstly, we need to find the mass of the nanocell.

>>> import h5py
>>> import numpy as np
>>> from micmec.analysis.advanced import get_mass, get_cell0, get_elasticity0
>>> f = h5py.File("example_path/example_trajectory.h5", "r")
>>> mass = get_mass(f)

Assuming the atomistic MD simulation was performed in the (N, P, T) ensemble, we can also find the equilibrium cell matrix and the elasticity tensor of the nanocell.

>>> cell = get_cell0(f)
>>> elasticity = get_elasticity0(f)

The start, end and sampling step of the trajectory can be modified to obtain a better estimate of the elastic properties.

Finally, we should note that the trajectory analysis routines included in MicMec are not limited to atomistis MD simulations. We can apply the same methods (``get_mass``, ``get_cell0`` and ``get_elasticity0``) to HDF5 files from micromechanical MD simulations, which is very useful for testing and valdiation.


PICKLE files
============

In MicMec, the properties of a cell are stored in a PICKLE file by default. In the ``data`` directory, four examples of PICKLE files are shown, including the scripts that were used to generate them.

-   ``data/test/types/generate_test_types.py``
    -   ``type_test.pickle``
    -   ``type_shrunk.pickle``
-   ``data/uio66/types/generate_fcu_type.py``
    -   ``type_fcu.pickle``
-   ``data/uio66/types/generate_reo_type.py``
    -   ``type_reo.pickle``

Please note that the **fcu** and **reo** type files cannot be generated without appropriate atomistic trajectory files. These atomistic trajectory files are multiple gigabytes in size and are therefore not included in MicMec.

PICKLE files are not human-readable, but they can be opened and read with Python easily, as follows.

>>> import pickle
>>> example_file = open("type_test.pickle",'rb')
>>> contents = pickle.load(example_file)
>>> print(contents)
>>> example_file.close()

Some nanocell types are multistable and therefore have multiple sets of elastic properties. The ``cell``, ``elasticity`` and ``free_energy`` groups of such a PICKLE file are always defined as lists to account for that fact.


