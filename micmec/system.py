#!/usr/bin/env python
# File name: system.py
# Description: Load, save, construct or edit a micromechanical system.
# Author: Joachim Vandewalle
# Date: 17-10-2021

"""Representation of a micromechanical system."""

import h5py
import numpy as np

from micmec.log import log, timer
from micmec.analysis.tensor import voigt

from yaff.pes.ext import Cell

from molmod.units import *
from molmod.io.chk import *


__all__ = ["System"]


class System(object):
    """Construct a micromechanical system.
    
    Parameters
    ----------
    pos : numpy.ndarray, shape=(``nnodes``, 3)
        The Cartesian coordinates of the micromechanical nodes.
    masses : numpy.ndarray, shape=(``nnodes``,)
        The masses of the micromechanical nodes.
    rvecs : numpy.ndarray, shape=(``nper``, 3)
        The domain vectors, which represent the periodicity of the simulation domain.

        -   ``nper = 3`` corresponds to periodic boundary conditions in the ``a``, ``b`` and ``c`` directions, where ``a``, ``b`` and ``c`` are the domain vectors.
        -   ``nper = 0`` corresponds to an absence of periodicity, used to simulate finite (three-dimensional) crystals.
        
    surrounding_cells : numpy.ndarray, dtype=int, shape=(``nnodes``, 8)
        The cells adjacent to each node.
        Each node has, at most, eight surrounding cells.
        In some cases, nodes have less than eight surrounding cells, due to internal surfaces (mesopores)
        or external surfaces (finite crystals).
        In that case, the specific order in which the surrounding cells are listed, is preserved, but the indices
        of the missing cells are replaced with -1.
    surrounding_nodes : numpy.ndarray, dtype=int, shape=(``ncells``, 8)
        The nodes adjacent to each cell.
        Each cell has eight surrounding nodes, always.
        The order in which the surrounding nodes of a given cell are listed, is specific and must not be changed.
    boundary_nodes : numpy.ndarray, dtype=int, optional
        The nodes at the boundary of the micromechanical system.
        The minimum image convention is only relevant for these boundary nodes.
    grid : numpy.ndarray, dtype=int, shape=(``nx``, ``ny``, ``nz``), optional
        A three-dimensional grid that maps the types of cells present in the micromechanical system.
        An integer value of 0 in the grid signifies an empty cell, a vacancy.
        An integer value of 1 signifies a cell of type 1, a value of 2 signifies a cell of type 2, etc.
    types : numpy.ndarray, dtype=int, shape=(``ncells``, 3), optional
        The cell types present in the micromechanical system.
    params : dict, optional
        The coarse-grained parameters of the micromechanical system. 
        A system may consist of several cell types, each with one or more metastable states.
        A single metastable state is represented by an equilibrium cell matrix, a free energy, and an elasticity tensor.
            
    Notes
    -----
    All quantities are expressed in atomic units.
    The optional arguments of the ``System`` class are technically not required to initialize a system, but all of
    them are required to perform a simulation.
    """
    
    def __init__(self, pos, masses, rvecs, surrounding_cells, surrounding_nodes,
                        boundary_nodes=None, grid=None, types=None, params=None):
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
            # Log some interesting system information.
            log.hline()
            pbc = (self.domain.rvecs.shape[0] > 0)
            gigapascal = 1e9*pascal
            if pbc:
                log("Periodic boundary conditions have been enabled.")
            else:
                log("Periodic boundary conditions have not been enabled. The system is isolated.")
            log(f"The system is composed of {self.nnodes} nodes and {self.ncells} cells.")
            ntypes = len(set(self.types))
            if ntypes == 1:
                log(f"There is {ntypes} cell type present in the system.")
            else:
                log(f"There are {ntypes} distinct cell types present in the system.")
            log(" ")
            for type_ in set(self.types):
                if int(type_) == 0:
                    # This should never happen, so maybe raise an exception?
                    continue
                h0 = self.params[f"type{int(type_)}/cell"]
                C0 = self.params[f"type{int(type_)}/elasticity"]
                efree = self.params[f"type{int(type_)}/free_energy"]
                eff = self.params[f"type{int(type_)}/effective_temp"]
                nstates = len(h0)
                if nstates == 1:
                    log(f"TYPE {type_} has {nstates} metastable state.")
                else:
                    log(f"TYPE {type_} has {nstates} metastable states.")
                for i in range(nstates):   
                    log(f"TYPE {type_}, STATE {i} : ") 
                    log(f"free energy [kj/mol] : ")
                    log(f"      {efree[i]/kjmol}")
                    log(f"equilibrium cell matrix [Å] :")
                    log("    [[{:6.1f}, {:6.1f}, {:6.1f}],".format(*list(h0[i][0]/angstrom)))
                    log("     [{:6.1f}, {:6.1f}, {:6.1f}],".format(*list(h0[i][1]/angstrom)))
                    log("     [{:6.1f}, {:6.1f}, {:6.1f}]]".format(*list(h0[i][2]/angstrom)))
                    log(f"elasticity tensor [GPa] :")
                    log("    [[{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(voigt(C0[i])[0]/gigapascal)))
                    log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(voigt(C0[i])[1]/gigapascal)))
                    log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(voigt(C0[i])[2]/gigapascal)))
                    log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(voigt(C0[i])[3]/gigapascal)))
                    log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(voigt(C0[i])[4]/gigapascal)))
                    log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}]]".format(*list(voigt(C0[i])[5]/gigapascal)))
                    log(" ")
            log.hline()
            log.blank()
    

    @classmethod
    def from_file(cls, fn, **user_kwargs):
        """Construct a micromechanical system from a CHK file.

        Parameters
        ----------
        fn : str
            The name of the CHK file to load.
        user_kwargs : dict, optional
            Additional information to construct the micromechanical system.

        Raises
        ------
        IOError
            If the input is not a CHK file.

        Returns
        -------
        micmec.system.System
            The micromechanical system, constructed from the CHK file.
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
        """Construct a micromechanical system from an HDF5 file with a system group.
        
        Parameters
        ----------
        f : h5py.File
            An HDF5 file with a system group. 
            The system group must at least contain a ``pos`` dataset.
        
        Returns
        -------
        micmec.system.System
            The micromechanical system, constructed from the HDF5 file.
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
        """Write the micromechanical system to a CHK, HDF5 or XYZ file.
        
        Parameters
        ----------
        fn : str
            The name of the file to write to.

        Raises
        ------
        NotImplementedError
            If the extension of the file does not match ``.chk``, ``.h5`` or ``.xyz``.

        Notes
        -----
        Supported file formats are:
        
        -   CHK (``.chk``) : internal text-based checkpoint format.
            This format includes all information about the ``System`` instance. 
        -   HDF5 (``.h5``) : internal binary checkpoint format. 
            This format includes all information about the ``System`` instance. 
        -   XYZ (``.xyz``) : a simple file with node positions.

        All data are stored in atomic units.
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
        """Write the system to an HDF5 file.
        
        Parameters
        ----------
        f : h5py.File
            A writable HDF5 file.

        Raises
        ------
        ValueError
            If the HDF5 file already contains a system description.
        """
        if "system" in f:
            raise ValueError("The HDF5 file already contains a system description.")
        sgrp = f.create_group("system")
        sgrp.create_dataset("pos", data=self.pos)
        if self.masses is not None:
            sgrp.create_dataset("masses", data=self.masses)


