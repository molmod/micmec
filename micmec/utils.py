#!/usr/bin/env python

#   MicMec 1.0, the first implementation of the micromechanical model, ever.
#               Copyright (C) 2022  Joachim Vandewalle
#                    joachim.vandewalle@hotmail.be
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#                  (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#              GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see https://www.gnu.org/licenses/.


"""Auxiliary routines for system construction."""

import numpy as np

from molmod.units import kelvin


__all__ = ["build_system", "build_type", "neighbor_cells", "neighbor_nodes", "Grid"]


# Each node has at most eight neighboring cells.
neighbor_cells = [
    (0, 0, 0),
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (-1, -1, 0),
    (-1, 0, -1),
    (0, -1, -1),
    (-1, -1, -1),
]
# Each cell always has eight neighboring nodes.
neighbor_nodes = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
]


class Grid(object):
    def __init__(self, grid, pbc=None):
        self.grid = grid

        if (pbc is None) or (pbc is True):
            self.pbc = [True, True, True]
        elif pbc is False:
            self.pbc = [False, False, False]
        else:
            self.pbc = pbc

        self.nx, self.ny, self.nz = grid.shape

        self.nx_nodes = self.nx + 1 - self.pbc[0]
        self.ny_nodes = self.ny + 1 - self.pbc[1]
        self.nz_nodes = self.nz + 1 - self.pbc[2]

        self._nodes = None
        self._boundary_nodes = None
        self._cells = None
        self._types = None

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._get_nodes()
        return self._nodes

    @property
    def nnodes(self):
        if self._nodes is None:
            self._nodes = self._get_nodes()
        return len(self._nodes)

    @property
    def boundary_nodes(self):
        if self._boundary_nodes is None:
            self._boundary_nodes = self._get_boundary_nodes()
        return self._boundary_nodes

    @property
    def cells(self):
        if self._cells is None:
            self._cells, self._types = self._get_cells_types()
        return self._cells

    @property
    def ncells(self):
        if self._cells is None:
            self._cells, self._types = self._get_cells_types()
        return len(self._cells)

    @property
    def types(self):
        if self._types is None:
            self._cells, self._types = self._get_cells_types()
        return self._types

    def _get_nodes(self):
        # Avoid nested for-loops by keeping the coordinates of the nodes in a 1D list.
        nodes = []
        for kk in range(self.nx_nodes):
            for ll in range(self.ny_nodes):
                for mm in range(self.nz_nodes):
                    for neighbor_idxs in neighbor_cells:
                        kappa = kk + neighbor_idxs[0]
                        lambda_ = ll + neighbor_idxs[1]
                        mu = mm + neighbor_idxs[2]
                        kappa_invalid = (kappa < 0 or kappa >= self.nx) and (
                            not self.pbc[0]
                        )
                        lambda_invalid = (lambda_ < 0 or lambda_ >= self.ny) and (
                            not self.pbc[1]
                        )
                        mu_invalid = (mu < 0 or mu >= self.nz) and (not self.pbc[2])
                        if kappa_invalid or lambda_invalid or mu_invalid:
                            continue
                        # If any surrounding nanocell is non-empty, then the node is valid.
                        if self.grid[(kappa, lambda_, mu)] != 0:
                            nodes.append((kk, ll, mm))
                            break
        return nodes

    def _get_boundary_nodes(self):
        # Store the indices of nodes located at the boundaries of the grid.
        boundary_nodes = []
        for node_idx, (kk, ll, mm) in enumerate(self.nodes):
            bx = kk == 0 or kk == self.nx_nodes - 1
            by = ll == 0 or ll == self.ny_nodes - 1
            bz = mm == 0 or mm == self.nz_nodes - 1
            if bx or by or bz:
                boundary_nodes.append(node_idx)
        return boundary_nodes

    def _get_cells_types(self):
        # Construct a list that contains the type of each nanocell.
        # Avoid nested for-loops by keeping the coordinates of the cells in a 1D list.
        cells = []
        types = []
        for kappa in range(self.nx):
            for lambda_ in range(self.ny):
                for mu in range(self.nz):
                    # Ensure that there is a cell present at (kappa, lambda, mu).
                    if self.grid[(kappa, lambda_, mu)] != 0:
                        cells.append((kappa, lambda_, mu))
                        types.append(self.grid[(kappa, lambda_, mu)])
        return cells, types


def build_system(data, grid, pbc=None):
    """Prepare a micromechanical system and store it in a dictionary.

    Parameters
    ----------
    data : dict
        The micromechanical cell types, stored in a dictionary with integer keys.
        The corresponding values are dictionaries which contain information about the cell type.
    grid : numpy.ndarray, dtype=int, shape=(``nx``, ``ny``, ``nz``)
        A three-dimensional grid that maps the types of cells present in the micromechanical system.
        An integer value of 0 in the grid signifies an empty cell, a vacancy.
        An integer value of 1 signifies a cell of type 1, a value of 2 signifies a cell of type 2, etc.
    pbc : list of bool, default=[True, True, True], optional
        The domain vectors for which periodic boundary conditions should be enabled.

    Returns
    -------
    output : dict
        A dictionary which is ready to be stored as a CHK file, containing a complete description of the micromechanical system.

    Notes
    -----
    This method is also used by the Micromechanical Model Builder, specifically by the ``builder_io.py`` script.
    If the Builder application does not work, the user can still benefit from the automatic creation
    of a micromechanical structure by using this method manually.
    The output dictionary can be stored as a CHK file by using:

        ``molmod.io.chk.dump_chk("output.chk", output)``.

    Then, the CHK file can be used as the input of a ``micmec.system.System`` instance.
    """
    grid_structure = Grid(grid, pbc=pbc)

    # Construct masses of the nodes and neighbor lists for both nanocells and nodes.
    masses = np.zeros(grid_structure.nnodes)
    surrounding_nodes = -np.ones((grid_structure.ncells, 8), dtype=int)
    surrounding_cells = -np.ones((grid_structure.nnodes, 8), dtype=int)
    # Iterate over each cell.
    for cell_idx, cell_idxs in enumerate(grid_structure.cells):
        kappa = cell_idxs[0]
        lambda_ = cell_idxs[1]
        mu = cell_idxs[2]
        type_idx = grid[(kappa, lambda_, mu)]
        type_data = data[type_idx]
        # Iterate over the eight neighboring nodes of the cell.
        for neighbor_idx, neighbor_idxs in enumerate(neighbor_nodes):
            kk = (kappa + neighbor_idxs[0]) % grid_structure.nx_nodes
            ll = (lambda_ + neighbor_idxs[1]) % grid_structure.ny_nodes
            mm = (mu + neighbor_idxs[2]) % grid_structure.nz_nodes
            for node_idx, node_idxs in enumerate(grid_structure.nodes):
                if kk == node_idxs[0] and ll == node_idxs[1] and mm == node_idxs[2]:
                    surrounding_nodes[cell_idx, neighbor_idx] = node_idx
                    surrounding_cells[node_idx, neighbor_idx] = cell_idx
                    masses[node_idx] += 0.125 * type_data["mass"]
                    break

    # Construct the initial positions of the nodes.
    pos = np.zeros((grid_structure.nnodes, 3))
    # Get majority type in the grid.
    maj_type_idx = np.argmax(np.bincount(np.array(grid.flatten(), dtype=int))[1:]) + 1
    maj_type_data = data[maj_type_idx]
    # Get dimensions of majority type cell.
    dx, dy, dz = np.diag(maj_type_data["cell"][0])
    # Iterate over each node.
    for node_idx, node_idxs in enumerate(grid_structure.nodes):
        kk = node_idxs[0]
        ll = node_idxs[1]
        mm = node_idxs[2]
        # The initial positions of the nodes are those of a rectangular grid.
        pos[node_idx, :] += np.array([kk * dx, ll * dy, mm * dz])

    # Build the rvecs array, which has either a (0, 3) or (1, 3) or (2, 3) or (3, 3) shape.
    rvecs = []
    if grid_structure.pbc[0]:
        rvecs.append(np.array([grid_structure.nx * dx, 0.0, 0.0]))
    if grid_structure.pbc[1]:
        rvecs.append(np.array([0.0, grid_structure.ny * dy, 0.0]))
    if grid_structure.pbc[2]:
        rvecs.append(np.array([0.0, 0.0, grid_structure.nz * dz]))
    rvecs = np.array(rvecs)
    if rvecs.size == 0:
        rvecs = np.zeros((0, 3))

    # Build the output dictionary.
    output = {
        "pos": pos,
        "rvecs": rvecs,
        "types": grid_structure.types,
        "grid": grid,
        "pbc": grid_structure.pbc,
        "surrounding_cells": surrounding_cells,
        "surrounding_nodes": surrounding_nodes,
        "boundary_nodes": grid_structure.boundary_nodes,
        "masses": masses,
    }
    for type_idx, type_data in data.items():
        for key, value in type_data.items():
            output[f"type{type_idx}/{key}"] = value

    return output


def build_type(
    material,
    mass,
    cell0,
    elasticity0,
    free_energy=None,
    effective_temp=None,
    topology=None,
):
    """Prepare a micromechanical cell type and store it in a dictionary.

    Parameters
    ----------
    material : str
        The name of the cell type's material.
    mass : float
        The total mass of the cell type.
    cell0 : (list of) numpy.ndarray, shape=(3, 3)
        The equilibrium cell matrix for each metastable state of the cell type.
    elasticity0 : (list of) numpy.ndarray, shape=(3,3,3,3)
        The elasticity tensor for each metastable state of the cell type.
    free_energy : (list of) float, optional
        The free energy for each metastable state of the cell type.
    effective_temp : float, optional
        The effective temperature for a multistable cell type.
    topology : str, optional
        The topology of the cell type's atomic structure.

    Raises
    ------
    ValueError
        If the number of metastable states is not consistent for the equilibrium cell matrix, elasticity tensor or free energy.

    Returns
    -------
    output : dict
        A dictionary which is ready to be stored as a PICKLE file, containing a complete description of the micromechanical cell type.
    """
    if type(cell0) is not list:
        cell0 = [cell0]
    if type(elasticity0) is not list:
        elasticity0 = [elasticity0]
    if topology is None:
        topology = "UNKNOWN"
    if effective_temp is None:
        effective_temp = 300 * kelvin
    if free_energy is None:
        free_energy = [0.0] * len(cell0)
    if type(free_energy) is not list:
        free_energy = [free_energy]
    output = {
        "material": material,
        "topology": topology,
        "mass": mass,
        "cell": cell0,
        "elasticity": elasticity0,
        "free_energy": free_energy,
        "effective_temp": effective_temp,
    }
    check1 = len(cell0) == len(elasticity0)
    check2 = len(cell0) == len(free_energy)
    if check1 and check2:
        return output
    raise ValueError
