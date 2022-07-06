#!/usr/bin/env python
# File name: system.py
# Description: Load, save, construct or edit a micromechanical system.
# Author: Joachim Vandewalle
# Date: 17-10-2021

"""Load, save, construct or edit a micromechanical system for use in simulations."""

import h5py
import numpy as np
import pickle as pkl

from micmec.log import log, timer

from yaff.pes.ext import Cell

from molmod.units import *
from molmod.io.chk import *

__all__ = ["System"]

class System(object):
    
    def __init__(self, pos, masses, rvecs, surrounding_cells, surrounding_nodes,
                        boundary_nodes=None, grid=None, types=None, params=None):
        """Construct a micromechanical system for use in simulations.
        
        Parameters
        ----------
        pos : 
            SHAPE: (nnodes, 3) 
            TYPE: numpy.ndarray
            DTYPE: float
            The Cartesian coordinates of the micromechanical nodes.
        masses :
            SHAPE: (nnodes,) 
            TYPE: numpy.ndarray
            DTYPE: float
            The masses of the micromechanical nodes.
        rvecs :
            SHAPE: (ndim, 3) 
            TYPE: numpy.ndarray
            DTYPE: float
            The Cartesian domain vectors, which represent the periodicity of the simulation domain.
            In the current version of MicMec, only ndim = 3 and ndim = 0 are supported.
            ndim = 3 corresponds to periodic boundary conditions in the a, b and c directions, where a, b and c
            are the Cartesian domain vectors.
            ndim = 0 corresponds to an absence of periodicity, used to simulate finite (three-dimensional) crystals.
        surrounding_cells :
            SHAPE: (nnodes, 8) 
            TYPE: numpy.ndarray
            DTYPE: int
            The cells adjacent to each node.
            Each node has, at most, eight surrounding cells.
            In some cases, nodes have less than eight surrounding cells, due to internal surfaces (mesopores)
            or external surfaces (finite crystals).
            In that case, the specific order in which the surrounding cells are listed, is preserved, but the indices
            of the missing cells are replaced with -1.
            Example:
                surrounding_cells[13] = np.array([-1, 4, 10, 12, 1, 3, 9, 0])
                # node 13 is surrounded by an empty cell (-1) and cells 4, 10, 12, 1, 3, 9 and 0
        surrounding_nodes :
            SHAPE: (ncells, 8) 
            TYPE: numpy.ndarray
            DTYPE: int
            The nodes adjacent to each cell.
            Each cell has eight surrounding nodes, always.
            The order in which the surrounding nodes of a given cell are listed, is specific and must not be changed.
            (Please refer to neighbors.py for a discussion of the order.)
            Example: 
                surrounding_nodes[0] = np.array([0, 9, 3, 1, 12, 10, 4, 13])
                # cell 0 is surrounded by nodes 0, 9, 3, 1, 12, 10, 4 and 13  
        boundary_nodes : optional
            TYPE: numpy.ndarray
            DTYPE: int
            The nodes at the boundary of the micromechanical system.
            The minimum image convention is only relevant for these boundary nodes.
        grid : optional
            SHAPE: (nx, ny, nz) 
            TYPE: numpy.ndarray
            DTYPE: int
            A three-dimensional grid that maps the types of cells present in the micromechanical system.
            An integer value of 0 in the grid signifies an empty cell, a vacancy.
            An integer value of 1 signifies a cell of type 1, etc.
        types : optional
            SHAPE: (ncells, 3) 
            TYPE: numpy.ndarray
            DTYPE: int
            The types of cells present in the micromechanical system.
            Example:
                types[13] = 1 # cell 13 belongs to type 1
        params : dict, optional
            The parameters of the micromechanical system.
            Example:
                params["type1/cell"] = np.array([[...], [...], [...]]) # equilibrium cell matrix of type 1
                
        Notes
        -----
        All quantities are expressed in atomic units.
        The optional arguments of the `System` class are technically not required to initialize a system, but all of
        them are required to perform a simulation. 
        
        """
        self.masses = masses
        self.pos = pos
        self.domain = Cell(rvecs) # yaff.pes.ext.Cell
        self.grid = grid
        self.types = types
        self.params = params
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
        """Create a micromechanical system from a .chk file.

        Parameters
        ----------
        fn : str
            The name of the .chk file to load.
        user_kwargs : dict, optional
            Keyword arguments used to create the micromechanical system.

        Raises
        ------
        IOError
            If the input is not a .chk file.

        Returns
        -------
        micmec.system.System object
            The micromechanical system, created from the .chk file.

        """
        with log.section("SYS"):
            kwargs = {}
            params = {}
            if fn.endswith(".chk"):
                allowed_keys = ["pos", 
                                "masses", 
                                "rvecs", 
                                "surrounding_cells", 
                                "surrounding_nodes", 
                                "boundary_nodes", 
                                "grid", 
                                "types"]
                for key, value in load_chk(fn).items():
                    if key in allowed_keys:
                        kwargs.update({key: value})
                    if "type" in key and key != "types":
                        params.update({key: value})
            else:
                raise IOError("Cannot read from file \"%s\"." % fn)
            if log.do_high:
                log("Read system parameters from %s." % fn)
            kwargs.update({"params": params})
            kwargs.update(user_kwargs)
        return cls(**kwargs)

    
    @classmethod
    def from_hdf5(cls, f):
        """Create a micromechanical system from a .h5 file with a system group.
        
        Parameters
        ----------
        f : h5py.File object (open)
            A .h5 file with a system group. 
            The system group must at least contain a `pos` dataset.
        
        Returns
        -------
        micmec.system.System object
            The micromechanical system, created from the .h5 file.
        
        """
        sgrp = f["system"]
        kwargs = {
            "pos": sgrp["pos"][:],
        }
        allowed_keys = [
            "pos", 
            "masses", 
            "rvecs", 
            "surrounding_cells", 
            "surrounding_nodes", 
            "boundary_nodes", 
            "grid", 
            "types", 
            "params"
        ]
        for key in allowed_keys:
            if key in sgrp:
                kwargs[key] = sgrp[key][:]
        if log.do_high:
            log("Read system parameters from %s." % f.filename)
        return cls(**kwargs)

    
    def to_file(self, fn):
        """Write the micromechanical system to a .h5 or .xyz file.
        
        Parameters
        ----------
        fn : str
            The name of the file to write to.

        Raises
        ------
        NotImplementedError
            If the extension of the file does not match `.chk`, `.h5` or `.xyz`.

        Notes
        -----
        Supported file formats are:
        .chk
            Internal text-based checkpoint format.
            This format includes all the information of the System object. 
            All data are stored in atomic units.
        .h5
            Internal binary checkpoint format. 
            This format includes all the information of the System object. 
            All data are stored in atomic units.
        .xyz
            A simple file with node positions.

        """
        if fn.endswith('.chk'):
            from molmod.io import dump_chk
            output = {
                "pos": self.pos, 
                "masses": self.masses, 
                "rvecs": self.rvecs, 
                "surrounding_cells": self.surrounding_cells, 
                "surrounding_nodes": self.surrounding_nodes, 
                "boundary_nodes": self.boundary_nodes, 
                "grid": self.grid, 
                "types": self.types
            }
            output.update(self.params)
            dump_chk(fn, output)
        elif fn.endswith(".xyz"):
            from molmod.io import XYZWriter
            from molmod.periodic import periodic
            xyz_writer = XYZWriter(fn, [periodic[55].symbol for _ in self.pos]) # represent nodes with cesium atoms
            xyz_writer.dump(str(self), self.pos)
        elif fn.endswith(".h5"):
            with h5py.File(fn, "w") as f:
                self.to_hdf5(f)
        else:
            raise NotImplementedError("The extension of %s does not correspond to any known format." % fn)
        if log.do_high:
            with log.section("SYS"):
                log("Wrote system to %s." % fn)

    
    def to_hdf5(self, f):
        """Write the system to a .h5 file.
        
        Parameters
        ----------
        f : h5py.File object (open)
            A writable .h5 file.

        Raises
        ------
        ValueError
            If the .h5 file already contains a system description.
        
        """
        if "system" in f:
            raise ValueError("The .h5 file already contains a system description.")
        sgrp = f.create_group("system")
        sgrp.create_dataset("pos", data=self.pos)
        if self.masses is not None:
            sgrp.create_dataset("masses", data=self.masses)


