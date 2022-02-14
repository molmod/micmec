#!/usr/bin/env python
# File name: system.py
# Description: The construction of a system of micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

""" The construction of a system of micromechanical nodes. """

import numpy as np
import pickle as pkl

from log import log, timer

from molmod.io.chk import *

__all__ = ["System"]


class System(object):
    
    def __init__(self, pos, pos_ref, cell_ref, masses,
                    equilibrium_cell_matrices, 
                    equilibrium_inv_cell_matrices,
                    elasticity_tensors,
                    surrounding_cells, surrounding_nodes,
                    grid=None, types=None):
        """
        **Arguments:**
        input_data
            A dictionary with the names of the micromechanical nanocell types as keys. The corresponding values are
            dictionaries which contain all of the relevant data about the cell type.
            Example: input_data["fcu"] = {"elasticity": [np.array([[[[...]]]])], "cell": ...}

        input_colors_types
            A dictionary with integer keys. These integers appear in the input_grid.
            The values corresponding to the keys are tuples of a color and the name of a type.
            Example: input_colors_types[1] = ("#0000FF", "fcu")

        input_grid
            An array containing integers, which refer to the types of micromechanical nanocells.
            Example: input_grid[kappa, lambda, mu] = 0 is an empty cell at (kappa, lambda, mu).
                     input_grid[kappa", lambda", mu"] = 1 is an fcu cell at (kappa", lambda", mu").
        """
        
        # Initialize system variables.
        self.masses = masses
        self.pos_ref = pos_ref
        self.cell_ref = cell_ref
        self.pos = pos

        self.equilibrium_cell_matrices = equilibrium_cell_matrices
        self.equilibrium_inv_cell_matrices = equilibrium_inv_cell_matrices
        self.elasticity_tensors = elasticity_tensors

        self.surrounding_cells = surrounding_cells
        self.surrounding_nodes = surrounding_nodes

        self.nnodes = len(self.surrounding_cells) 
        self.ncells = len(self.surrounding_nodes)
        
        with log.section("SYS"):
            self._init_log()

    
    def _init_log(self):
        if log.do_medium:
            log("PRINT INTERESTING SYSTEM INFORMATION HERE")
            log.hline()
            log.blank()
    

    @classmethod
    def from_file(cls, fn, **user_kwargs):

        with log.section("SYS"):
            kwargs = {}
            if fn.endswith(".chk"):

                allowed_keys = ["pos", "pos_ref", "cell_ref", "masses", 
                                "equilibrium_cell_matrices", 
                                "equilibrium_inv_cell_matrices",
                                "elasticity_tensors",
                                "surrounding_cells", "surrounding_nodes"]
                
                for key, value in load_chk(fn).items():
                    if key in allowed_keys:
                        kwargs.update({key: value})

            else:
                raise IOError("Cannot read from file \'%s\'." % fn)
            if log.do_high:
                log("Read system parameters from %s." % fn)
            kwargs.update(user_kwargs)
        
        return cls(**kwargs)

    
    def to_file(self, fn):
        
        """
        Write the system to a file.
        
        **Arguments:**
        fn
            The file to write to.

        Supported formats are:

        h5
            Internal binary checkpoint format. This format includes
            all the information of a system object. All data are stored in
            atomic units.

        xyz
            A simple file with node positions.

        """
        if fn.endswith(".xyz"):
            from molmod.io import XYZWriter
            from molmod.periodic import periodic
            xyz_writer = XYZWriter(fn, [periodic[55].symbol for _ in self.pos]) # represent nodes with cesium atoms
            xyz_writer.dump(str(self), self.pos)
        elif fn.endswith(".h5"):
            with h5.File(fn, "w") as f:
                self.to_hdf5(f)
        else:
            raise NotImplementedError("The extension of %s does not correspond to any known format." % fn)
        if log.do_high:
            with log.section("SYS"):
                log("Wrote system to %s." % fn)

    
    def to_hdf5(self, f):
        """
        Write the system to a HDF5 file.
        
        **Arguments:**
        
        f
            A writable h5.File object.
        
        """
        if "system" in f:
            raise ValueError("The HDF5 file already contains a system description.")
        sgrp = f.create_group("system")
        sgrp.create_dataset("pos", data=self.pos)
        if self.masses is not None:
            sgrp.create_dataset("masses", data=self.masses)


