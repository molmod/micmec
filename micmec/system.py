#!/usr/bin/env python
# File name: system.py
# Description: The construction of a system of micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

""" The construction of a system of micromechanical nodes. """

import numpy as np
import pickle as pkl

from .log import log, timer
from yaff.pes.ext import Cell

from molmod.units import *
from molmod.io.chk import *

__all__ = ["System"]


class System(object):
    
    def __init__(self, pos, masses, rvecs,
                    equilibrium_cell_matrices, 
                    equilibrium_inv_cell_matrices,
                    elasticity_tensors,
                    free_energies, effective_temps,
                    surrounding_cells, surrounding_nodes,
                    boundary_nodes,
                    grid=None, types=None):
        
        # Initialize system variables.
        self.masses = masses
        self.pos = pos
        self.domain = Cell(rvecs)

        self.grid = grid

        self.equilibrium_cell_matrices = equilibrium_cell_matrices
        self.equilibrium_inv_cell_matrices = equilibrium_inv_cell_matrices
        self.elasticity_tensors = elasticity_tensors
        
        #dt = 0.10*np.pi*np.sqrt(np.min(masses)/(np.max(equilibrium_cell_matrices)*np.max(elasticity_tensors)))
        #print("RECOMMENDED TIMESTEP: ", dt/femtosecond)

        self.free_energies = free_energies
        self.effective_temps = effective_temps

        self.surrounding_cells = surrounding_cells
        self.surrounding_nodes = surrounding_nodes

        self.boundary_nodes = boundary_nodes

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

                allowed_keys = ["pos", "masses", "rvecs",
                                "equilibrium_cell_matrices", 
                                "equilibrium_inv_cell_matrices",
                                "elasticity_tensors",
                                "free_energies",
                                "effective_temps",
                                "surrounding_cells", "surrounding_nodes",
                                "boundary_nodes", "grid"]
                
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
        
        **ARGUMENTS**
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
        
        **ARGUMENTS**
        f
            A writable h5.File object.
        
        """
        if "system" in f:
            raise ValueError("The HDF5 file already contains a system description.")
        sgrp = f.create_group("system")
        sgrp.create_dataset("pos", data=self.pos)
        if self.masses is not None:
            sgrp.create_dataset("masses", data=self.masses)


