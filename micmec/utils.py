#!/usr/bin/env python
# File name: utils.py
# Description: Auxiliary construction routines.
# Author: Joachim Vandewalle
# Date: 25-03-2022

"""Auxiliary routines for system construction."""

import numpy as np

__all__ = [
    "build_system", 
    "build_type"
]


def build_system(data, grid, pbc):
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
    pbc : list of bool
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
    
    nx, ny, nz = np.shape(grid)

    nx_nodes = nx + 1 - pbc[0]
    ny_nodes = ny + 1 - pbc[1]
    nz_nodes = nz + 1 - pbc[2]
    
    # Each node has at most eight neighboring cells.
    neighbor_cells = [
        ( 0, 0, 0),
        (-1, 0, 0),
        ( 0,-1, 0),
        ( 0, 0,-1),
        (-1,-1, 0),
        (-1, 0,-1),
        ( 0,-1,-1),
        (-1,-1,-1)
    ]
    # Each cell always has eight neighboring nodes.
    neighbor_nodes = [
        (0,0,0),
        (1,0,0),
        (0,1,0),
        (0,0,1),
        (1,1,0),
        (1,0,1),
        (0,1,1),
        (1,1,1)
    ]

    # Avoid nested for-loops by keeping the coordinates of the nodes in a 1D list.
    nodes = []
    for k in range(nx_nodes):
        for l in range(ny_nodes):
            for m in range(nz_nodes):
                for neighbor_idxs in neighbor_cells:
                    kappa = k + neighbor_idxs[0]
                    lambda_ = l + neighbor_idxs[1]
                    mu = m + neighbor_idxs[2]
                    if (not pbc[0]) and (kappa < 0 or kappa >= nx):
                        continue
                    if (not pbc[1]) and (lambda_ < 0 or lambda_ >= ny):
                        continue
                    if (not pbc[2]) and (mu < 0 or mu >= nz):
                        continue
                    # If any surrounding nanocell is non-empty, then the node is valid.
                    if grid[(kappa, lambda_, mu)] != 0:
                        nodes.append((k, l, m))
                        break
    nnodes = len(nodes)

    # Store the indices of nodes located at the boundaries of the grid.
    boundary_nodes = []
    for node_idx, (k, l, m) in enumerate(nodes):
        bx = (k == 0 or k == nx_nodes - 1)
        by = (l == 0 or l == ny_nodes - 1)
        bz = (m == 0 or m == nz_nodes - 1)
        if bx or by or bz:
            boundary_nodes.append(node_idx)
    
    # Construct a list that contains the type of each nanocell.
    # Avoid nested for-loops by keeping the coordinates of the cells in a 1D list.
    cells = []
    types = []
    for kappa in range(nx):
        for lambda_ in range(ny):
            for mu in range(nz):
                # Ensure that there is a cell present at (kappa, lambda, mu).
                if grid[(kappa, lambda_, mu)] != 0:
                    cells.append((kappa, lambda_, mu))
                    types.append(grid[(kappa, lambda_, mu)])
    ncells = len(cells)

    # Construct masses of the nodes and neighbor lists for both nanocells and nodes.
    masses = np.zeros(nnodes)
    surrounding_nodes = -np.ones((ncells, 8), dtype=int)
    surrounding_cells = -np.ones((nnodes, 8), dtype=int)
    # Iterate over each cell.
    for cell_idx, cell_idxs in enumerate(cells):
        kappa = cell_idxs[0]
        lambda_ = cell_idxs[1]
        mu = cell_idxs[2]
        type_idx = grid[(kappa, lambda_, mu)]
        type_data = data[type_idx]
        # Iterate over the eight neighboring nodes of the cell.
        for neighbor_idx, neighbor_idxs in enumerate(neighbor_nodes):
            k = (kappa + neighbor_idxs[0]) % nx_nodes
            l = (lambda_ + neighbor_idxs[1]) % ny_nodes
            m = (mu + neighbor_idxs[2]) % nz_nodes
            for node_idx, node_idxs in enumerate(nodes):
                if k == node_idxs[0] and l == node_idxs[1] and m == node_idxs[2]:
                    surrounding_nodes[cell_idx, neighbor_idx] = node_idx
                    surrounding_cells[node_idx, neighbor_idx] = cell_idx
                    masses[node_idx] += 0.125*type_data["mass"]
                    break 
    
    # Construct the initial positions of the nodes.
    pos = np.zeros((nnodes, 3))
    # Get majority type in the grid.
    maj_type_idx = np.argmax(np.bincount(np.array(grid.flatten(), dtype=int))[1:]) + 1
    maj_type_data = data[maj_type_idx]
    # Get dimensions of majority type cell.
    dx, dy, dz = np.diag(maj_type_data["cell"][0])
    # Iterate over each node.
    for node_idx, node_idxs in enumerate(nodes):
        k = node_idxs[0]
        l = node_idxs[1]
        m = node_idxs[2]
        # The initial positions of the nodes are those of a rectangular grid.
        pos[node_idx, :] += np.array([k*dx, l*dy, m*dz])
    
    # Build the rvecs array, which has either a (0, 3) or (1, 3) or (2, 3) or (3, 3) shape.
    rvecs = []
    if pbc[0]:
        rvecs.append(np.array([nx*dx, 0.0, 0.0]))
    if pbc[1]:
        rvecs.append(np.array([0.0, ny*dy, 0.0]))
    if pbc[2]:
        rvecs.append(np.array([0.0, 0.0, nz*dz]))
    rvecs = np.array(rvecs)
    if rvecs.size == 0:
        rvecs = np.zeros((0, 3))
    
    # Build the output dictionary.
    output = {}
    for type_idx, type_data in data.items():
        for key, value in type_data.items():
            output[f"type{type_idx}/{key}"] = value       
    output["pos"] = pos
    output["rvecs"] = rvecs
    output["types"] = types
    output["grid"] = grid
    output["pbc"] = pbc
    output["surrounding_cells"] = surrounding_cells
    output["surrounding_nodes"] = surrounding_nodes
    output["boundary_nodes"] = boundary_nodes
    output["masses"] = masses
    
    return output


def build_type():
    pass



