Constructing micromechanical systems
####################################

In Yaff, atomic structures are written to CHK files. These files are human-readable and are therefore ideal for quick reference and editing. In MicMec, we have attempted to replicate this experience. The following two directories contain examples of micromechanical structures, stored as CHK files:

-   ``data/test/struct/``,
-   ``data/uio66/struct/``.

The CHK files can be opened in a text editor to view their contents. Immediately, you should notice an important distinction between CHK files in Yaff and MicMec. Yaff stores atomic structures and force field parameters in separate files, while MicMec incorporates the micromechanical force field parameters (i.e. the cell types) into its micromechanical structure files. The advantage of Yaff is obvious: an atomic structure should not be tied to an atomistic force field, however accurate it may be. However, we should also note that the micromechanical force field parameters in MicMec's CHK files are included in separate groups, labeled according to cell type (``"type1/cell"``, ``"type1/elasticity"``...). Thus, the advantage of Yaff is redundant, because we can edit the micromechanical force field parameters separately from the structure parameters.

Constructing a micromechanical system from scratch is a difficult and time-consuming process. It can be particularly tedious to determine a reasonable estimate for the initial positions of the micromechanical nodes, for instance. Similarly, it can be very cumbersome to assign a cell type to every cell of a large micromechanical system, especially when there are thousands of cells. Luckily, much of the micromechanical construction process has been automated in MicMec.

We present two options to construct a micromechanical system fast and efficiently: a code-based approach and a GUI-based approach. The code-based approach relies solely on typing, while the GUI-based approach relies on clicking buttons in an application, the Micromechanical Model Builder. The Micromechanical Model Builder is our dedicated application for the design and visualisation of micromechanical systems.


Code-based construction
=======================

The only information we need to prepare micromechanical system, is given by the arguments of ``micmec.utils.build_system()``, which we explain here.

-   ``data`` : dict
     The micromechanical cell types, stored in a dictionary with integer keys. The corresponding values are dictionaries which contain information about the cell type (most importantly, equilibrium cell matrix, elasticity tensor and free energy of each metastable state).
-   ``grid`` : numpy.ndarray, dtype=int, shape=(``nx``, ``ny``, ``nz``)
     A three-dimensional grid that maps the types of cells present in the micromechanical system. An integer value of 0 in the grid signifies an empty cell, a vacancy. An integer value of 1 signifies a cell of type 1, a value of 2 signifies a cell of type 2, etc.
-   ``pbc`` : list of bool, default=[True, True, True]
     The domain vectors for which periodic boundary conditions should be enabled.

This method returns an ``output`` dictionary. The ``output`` dictionary can be dumped in a CHK file (``"output.chk"``), as follows, in a Python script.

>>> molmod.io.chk.dump_chk("output.chk", output)

That brings us full circle, to the formatting of CHK files in MicMec. The values of ``pos``, ``rvecs``, ``masses``, ``surrounding_nodes``, ``surrounding_cells`` and ``boundary_nodes`` have been calculated automatically in the ``build_system()`` method.

In summary, a code-based construction of a micromechanical system has the following steps.

#.  Define cell types by extracting the elastic properties of atomistic cells.
#.  Manually build a dictionary of all relevant cell types, named ``data``.
#.  Choose the locations of cell types in a three-dimensional ``grid``.
#.  Apply ``micmec.utils.build_system()``.
#.  Store the ``output`` dictionary in a CHK file, ``"output.chk"``.
#.  Apply ``micmec.system.System.from_file("output.chk")``.

The ``System`` instance, finally, can be used in simulations.


GUI-based construction: the Micromechanical Model Builder
=========================================================

To start the Micromechanical Model Builder, simply run the ``builder.py`` script.

.. code:: bash

   python micmec/builder/builder.py

Alternatively, if you have actually *installed* MicMec, you should be able to call the application with a command.

.. code:: bash

   micmec_builder

This application uses ``tkinter``, a built-in GUI package for Python. In fact, it only relies on built-in Python packages. In the top menubar of the application, please navigate to:

   ``Help > Tutorial``.

There, you should find a tutorial on how to use the application, complete with pictures. If you are impatient, you can navigate to:

   ``File > Load``,

where you can load a pre-existing CHK file. As mentioned previously, there are some CHK files included in MicMec, in the ``data`` directory. In the same directory, you can find PICKLE files, which contain individual cell types.

