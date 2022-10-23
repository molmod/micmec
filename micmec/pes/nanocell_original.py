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


"""Micromechanical description of a single nanocell state (original).

This script contains the original equations of the micromechanical model, as proposed by S. M. J. Rogge.
It is, however, NOT recommended to use these equations, as they are based on wrong assumptions.
Use the default script (``nanocell.py``) instead.
"""

import numpy as np

from micmec.pes.nanocell_utils import multiplicator, cell_xderivs, cell_yderivs, cell_zderivs

__all__ = ["elastic_energy_nanocell", "grad_elastic_energy_nanocell"]

# In the original model, the concept of `representations` does not exist.
# Therefore, we only need a few of these derivatives.
cell_xderiv = [0.25*xderiv[i] for i, xderiv in enumerate(cell_xderivs)]
cell_yderiv = [0.25*yderiv[i] for i, yderiv in enumerate(cell_yderivs)]
cell_zderiv = [0.25*zderiv[i] for i, zderiv in enumerate(cell_zderivs)]


def elastic_energy_nanocell(vertices, h0, C0):
    """The elastic deformation energy of a nanocell, with respect to one of its metastable states with parameters h0 and C0.
        
    Parameters
    ----------
    vertices : numpy.ndarray, shape=(8, 3)
        The coordinates of the surrounding nodes (i.e. the vertices).
    h0 : numpy.ndarray, shape=(3, 3)   
        The equilibrium cell matrix.
    C0 : numpy.ndarray, shape=(3, 3, 3, 3)
        The elasticity tensor.

    Returns
    -------
    energy : float
        The elastic deformation energy.
    
    Notes
    -----
    At first sight, the equations for bistable nanocells might seem absent from this derivation.
    They are absent here, but they have been implemented in the ``mmff.py`` script.
    This elastic deformation energy is only the energy of a single metastable state of a nanocell.
    """
    h0_inv = np.linalg.inv(h0) # [3x3]
    h0_det = np.linalg.det(h0)
    
    matrices = np.einsum("...i,ij->...j", multiplicator, vertices) # [8x3x8].[8x3] = [8x3x3]
    matrix = 0.125*np.einsum("i...->...", matrices) # [3x3]
    matrix_ = np.einsum("ji,kj->ik", matrix, h0_inv) # [3x3]

    identity_ = np.identity(3) # [3x3]
    strain = 0.5*(np.einsum("ji,jk->ik", matrix_, matrix_) - identity_) # [3x3]
    
    energy_density = 0.5*np.einsum("ij,ijkl,kl", strain, C0, strain)
    energy = energy_density*h0_det
    
    return energy


def grad_elastic_energy_nanocell(vertices, h0, C0):
    """The gradient of the elastic deformation energy of a nanocell (with respect to one of its metastable states with parameters 
    h0 and C0), towards the Cartesian coordinates of its surrounding nodes.
        
    Parameters
    ----------
    vertices : numpy.ndarray, shape=(8, 3)
        The coordinates of the surrounding nodes (i.e. the vertices).
    h0 : numpy.ndarray, shape=(3, 3)   
        The equilibrium cell matrix.
    C0 : numpy.ndarray, shape=(3, 3, 3, 3)
        The elasticity tensor.

    Returns
    -------
    gpos_state : numpy.ndarray, shape=(8, 3)
        The gradient of the elastic deformation energy (of a single state of the nanocell).
    """
    h0_inv = np.linalg.inv(h0) # [3x3]
    h0_det = np.linalg.det(h0)
    
    matrices = np.einsum("...i,ij->...j", multiplicator, vertices) # [8x3x8].[8x3] = [8x3x3]
    matrix = 0.125*np.einsum("i...->...", matrices) # [3x3]
    matrix_ = np.einsum("ji,kj->ik", matrix, h0_inv) # [3x3]
    
    xmat = np.einsum("ki,...km,jm->...ij", matrix_, cell_xderiv, h0_inv) # [3x3].[[8x3x3].[3x3]] = [8x3x3]
    ymat = np.einsum("ki,...km,jm->...ij", matrix_, cell_yderiv, h0_inv) 
    zmat = np.einsum("ki,...km,jm->...ij", matrix_, cell_zderiv, h0_inv)
    eps_xderiv = 0.5*(np.einsum("...ji", xmat) + xmat) # [8x3x3]
    eps_yderiv = 0.5*(np.einsum("...ji", ymat) + ymat) 
    eps_zderiv = 0.5*(np.einsum("...ji", zmat) + zmat)
    
    identity_ = np.identity(3) # [3x3]
    strain = 0.5*(np.einsum("ji,jk->ik", matrix_, matrix_) - identity_) # [3x3]
    
    stress = np.einsum("ijkl,kl->ij", C0, strain) # [3x3]
    gpos_state = np.zeros((8, 3))
    gpos_state[:, 0] += np.einsum("...ji,ij", eps_xderiv, stress) # [8x3x3].[3x3] = [8]
    gpos_state[:, 1] += np.einsum("...ji,ij", eps_yderiv, stress)
    gpos_state[:, 2] += np.einsum("...ji,ij", eps_zderiv, stress)
    
    return h0_det*gpos_state
