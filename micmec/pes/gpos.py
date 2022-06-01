#!/usr/bin/env python
# File name: mmf.py
# Description: MicroMechanicalField. The calculation of forces acting on the micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

"""The calculation of forces acting on the nodes of the micromechanical model."""

import numpy as np
import jax.numpy as jnp

from molmod import boltzmann
from jax import jit, grad

from ..log import log, timer

from time import time

__all__ = [
    "MicMecForceField", 
    "ForcePart", 
    "ForcePartMechanical", 
    "ForcePartPressure"
]


def _compute_cell_properties(self):
    """Compute the cell properties."""
    # Initialize an empty list to store the cell matrices and their inverses in.
    self.type_grads = []

    for params in type_params:
        self.types
    
    self.cell_mats = []
    self.cell_invs = []
    self.cell_dets = []
    self.cell_strains = []
    self.cell_epots = []
    self.cell_weights = []
    self.cell_verts = []
    
    for cell_idx in range(self.system.ncells):
        # Store the nodal index of each vertex of the current cell in an array.
        vertices = np.zeros((8,), int)
        for neighbor_idx, node_idx in enumerate(self.system.surrounding_nodes[cell_idx]):
            vertices[neighbor_idx] = node_idx
        # Store each edge vector of the current cell in an array.
        edges = np.zeros((12, 3), float)
        # Edges pointing in the x-direction.
        edges[0] = self.delta(vertices[0], vertices[1])
        edges[1] = self.delta(vertices[2], vertices[4])
        edges[2] = self.delta(vertices[3], vertices[5])
        edges[3] = self.delta(vertices[6], vertices[7])
        if self.system.grid.shape[0] == 2:
            for k in range(0, 4):
                if edges[k, 0] < 0.0:
                    edges[k] += self.system.domain.rvecs[0] 
        # Edges pointing in the y-direction.            
        edges[4] = self.delta(vertices[0], vertices[2])
        edges[5] = self.delta(vertices[1], vertices[4])
        edges[6] = self.delta(vertices[3], vertices[6])
        edges[7] = self.delta(vertices[5], vertices[7])
        if self.system.grid.shape[1] == 2:
            for k in range(4, 8):
                if edges[k, 1] < 0.0:
                    edges[k] += self.system.domain.rvecs[1]
        # Edges pointing in the z-direction.
        edges[8] = self.delta(vertices[0], vertices[3])
        edges[9] = self.delta(vertices[2], vertices[6])
        edges[10] = self.delta(vertices[1], vertices[5])
        edges[11] = self.delta(vertices[4], vertices[7])
        if self.system.grid.shape[2] == 2:
            for k in range(8, 12):
                if edges[k, 2] < 0.0:
                    edges[k] += self.system.domain.rvecs[2]
        # Construct each possible cell matrix of the current cell.
        h0 = np.array([edges[0], edges[4], edges[8]]).T
        h1 = np.array([edges[0], edges[5], edges[10]]).T
        h2 = np.array([edges[1], edges[4], edges[9]]).T
        h3 = np.array([edges[2], edges[6], edges[8]]).T
        h4 = np.array([edges[1], edges[5], edges[11]]).T
        h5 = np.array([edges[2], edges[7], edges[10]]).T
        h6 = np.array([edges[3], edges[6], edges[9]]).T
        h7 = np.array([edges[3], edges[7], edges[11]]).T
        # Calculate the position of each vertex.
        r0 = self.system.pos[vertices[0]]
        r1 = r0 + edges[0]
        r2 = r0 + edges[4]
        r3 = r0 + edges[8]
        r4 = r0 + edges[1] + edges[4]
        r5 = r0 + edges[2] + edges[8]
        r6 = r0 + edges[6] + edges[8]
        r7 = r0 + edges[11] + edges[1] + edges[4]
        # Construct the cell properties:
        # the cell matrix, the determinant of the cell matrix (i.e. the cell vollume)
        # and the inverse cell matrix.
        cell_mat = np.array([h0, h1, h2, h3, h4, h5, h6, h7])
        cell_det = np.linalg.det(cell_mat)
        cell_inv = np.linalg.inv(cell_mat)
        cell_vert = np.array([r0, r1, r2, r3, r4, r5, r6, r7])
        # Load the equilibrium cell properties.
        # Each metastable state has a different equilibrium inverse cell matrix, elasticity tensor
        # and free energy. The effective temperature is an additional fitting parameter.
        cell0_eff_temp = self.system.effective_temps[cell_idx]
        # Initialize the list of strain tensors and potential energies for each metastable state.
        cell_strain_lst = []
        cell_epot_lst = []
        # Iterate over each metastable state.
        for state_idx in zip(self.system.equilibrium_inv_cell_matrices[cell_idx], 
                                                        self.system.elasticity_tensors[cell_idx], 
                                                        self.system.free_energies[cell_idx]):
            cell_strain = self.strain(cell_mat, cell0_inv)
            cell_epot = self.elastic_energy(cell_strain, cell0_elast, cell_det)
            cell_strain_lst.append(cell_strain)
            cell_epot_lst.append(cell_epot + cell0_efree)
        cell_epot_lst = np.array(cell_epot_lst)
        cell_epot_min = np.min(cell_epot_lst)
        cell_weight_lst = np.exp(-(cell_epot_lst - cell_epot_min)/(boltzmann*cell0_eff_temp))
        cell_epot = cell_epot_min - cell0_eff_temp*boltzmann*np.log(np.sum(cell_weight_lst))

        # Store the cell matrices, the inverse cell matrices, the determinants and the strain tensors.
        self.cell_mats.append(cell_mat)
        self.cell_invs.append(cell_inv)
        self.cell_dets.append(cell_det)
        self.cell_strains.append(cell_strain_lst)
        self.cell_epots.append(cell_epot)
        self.cell_weights.append(cell_weight_lst)
        self.cell_verts.append(cell_vert)
    return None






@jit
def elastic_energy(vertices_flat, h0_inv, C0):
    """Compute the elastic energy of a nanocell."""
    vertices = vertices_flat.reshape((8, 3))
    # Store each edge vector of the current cell in an array.
    edges = jnp.array([
        vertices[1] - vertices[0],
        vertices[4] - vertices[2],
        vertices[5] - vertices[3],
        vertices[7] - vertices[6],         
        vertices[2] - vertices[0],
        vertices[4] - vertices[1],
        vertices[6] - vertices[3],
        vertices[7] - vertices[5],
        vertices[3] - vertices[0],
        vertices[6] - vertices[2],
        vertices[5] - vertices[1],
        vertices[7] - vertices[4]
    ])
    matrices = jnp.array([
        [edges[0], edges[4], edges[8] ],
        [edges[0], edges[5], edges[10]],
        [edges[1], edges[4], edges[9] ],
        [edges[2], edges[6], edges[8] ],
        [edges[1], edges[5], edges[11]],
        [edges[2], edges[7], edges[10]],
        [edges[3], edges[6], edges[9] ],
        [edges[3], edges[7], edges[11]]
    ])
    matrices_ = jnp.einsum("...ji,jk->...ik", matrices, h0_inv)
    identity_ = jnp.array([np.identity(3) for _ in range(8)])
    strains = 0.5*(jnp.einsum("...ji,...jk->...ik", matrices_, matrices_) - identity_)

    energy_density = 0.5*jnp.einsum("..ij,ijkl,..kl", strains, C0, strains)
    energy = 0.125*jnp.einsum("i,i", jnp.linalg.det(matrices), energy_density)

    return energy



def strain(h, h0_inv):
    """
    Mechanical strain.  
    h   
        cell matrix
    h0_inv
        inverse equilibrium cell matrix
    """
    
    
    return 0.5*(np.einsum("...ji,...jk->...ik", mat, mat) - iden)


@staticmethod
def force(h, h_xderiv, h_yderiv, h_zderiv, h0_inv, C0, eps=None, h_det=None, h_inv=None):
    #    8x3x3,8x3x3,    8x3x3,    8x3x3,    3x3,  3x3x3x3, 8x3x3, 8,          8x3x3
    """
    Micromechanical force.
    h
        cell matrix
    h_xderiv, h_yderiv, h_zderiv
        partial derivatives of cell matrix to the cartesian coordinates of a node
    h0_inv
        inverse equilibrium cell matrix
    C0
        (equilibrium) elasticity tensor
    eps
        strain tensor
    h_det
        determinant of cell matrix (volume of cell)
    h_inv
        inverse cell matrix
    """
    f = np.zeros((3, 8))
    if eps is None:
        eps = strain(h, h0_inv)
    if h_det is None:
        h_det = np.linalg.det(h)
    if h_inv is None:
        h_inv = np.linalg.inv(h)
    
    h_xtrace = np.einsum("...ij,...ji", h_inv, h_xderiv) #8
    h_ytrace = np.einsum("...ij,...ji", h_inv, h_yderiv)
    h_ztrace = np.einsum("...ij,...ji", h_inv, h_zderiv)

    mat = np.einsum("...ij,jk->...ki", h, h0_inv)
    xmat = np.einsum("...ij,...jk->...ik", mat, np.einsum("...ij,jk->...ik", h_xderiv, h0_inv)) #8x3x3
    ymat = np.einsum("...ij,...jk->...ik", mat, np.einsum("...ij,jk->...ik", h_yderiv, h0_inv))
    zmat = np.einsum("...ij,...jk->...ik", mat, np.einsum("...ij,jk->...ik", h_zderiv, h0_inv))
    
    eps_xderiv = 0.5*(np.einsum("...ji", xmat) + xmat)
    eps_yderiv = 0.5*(np.einsum("...ji", ymat) + ymat)
    eps_zderiv = 0.5*(np.einsum("...ji", zmat) + zmat)

    stress = np.einsum("ijkl,...kl->...ij", C0, eps) #8x3x3
    quad_form = np.einsum("...ij,...ij", eps, stress) #8
    
    # Compute the contribution of cell (kappa, lambda, mu) to the x component
    # of the force acting on node (k, l, m).
    xterm = np.einsum("...ji,...ij", eps_xderiv, stress) #8
    f[0] += h_xtrace*quad_form
    f[0] += 2.0*xterm
    
    # Compute the contribution of cell (kappa, lambda, mu) to the y component
    # of the force acting on node (k, l, m).
    yterm = np.einsum("...ji,...ij", eps_yderiv, stress)
    f[1] += h_ytrace*quad_form
    f[1] += 2.0*yterm
    
    # Compute the contribution of cell (kappa, lambda, mu) to the y component
    # of the force acting on node (k, l, m).
    zterm = np.einsum("...ji,...ij", eps_zderiv, stress)
    f[2] += h_ztrace*quad_form
    f[2] += 2.0*zterm
    
    # Scale the contribution of cell (kappa, lambda, mu) according to the volume of the cell.
    return -0.0625*np.einsum("i,...i", h_det, f)


