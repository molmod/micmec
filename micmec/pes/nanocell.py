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


"""Micromechanical description of a single nanocell state.

A single state of a nanocell, on the micromechanical level, is described by means of its elastic deformation energy and the gradient 
of that energy, which represents the forces acting on the micromechanical nodes.
"""
# In the comments, we refer to equations in the master's thesis of Joachim Vandewalle.

import numpy as np

from micmec.pes.nanocell_utils import multiplicator, cell_xderivs, cell_yderivs, cell_zderivs

__all__ = ["elastic_energy_nanocell", "grad_elastic_energy_nanocell"]


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
    # (3.20)
    matrices = np.einsum("...i,ij->...j", multiplicator, vertices) # [8x3x8].[8x3] = [8x3x3]

    # (3.23)
    matrices_ = np.einsum("...ji,kj->...ik", matrices, np.linalg.inv(h0))
    identity_ = np.array([np.identity(3) for _ in range(8)])
    strains = 0.5*(np.einsum("...ji,...jk->...ik", matrices_, matrices_) - identity_)
    
    # (4.11)
    energy_density = 0.5*np.einsum("...ij,ijkl,...kl", strains, C0, strains)
    energy = 0.125*np.einsum("i->", energy_density)*np.linalg.det(h0)
    
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
    
    # (3.20)
    matrices = np.einsum("...i,ij->...j", multiplicator, vertices) # [8x3x8].[8x3] = [8x3x3]
    matrices_ = np.einsum("...ji,kj->...ik", matrices, h0_inv) # [8x3x3]
    
    # (3.31)
    xmat = np.einsum("...ki,...km,jm->...ij", matrices_, cell_xderivs, h0_inv) # [8x3x3].[[8x8x3x3].[3x3]] = [8x8x3x3]
    ymat = np.einsum("...ki,...km,jm->...ij", matrices_, cell_yderivs, h0_inv) 
    zmat = np.einsum("...ki,...km,jm->...ij", matrices_, cell_zderivs, h0_inv)
    eps_xderiv = 0.5*(np.einsum("...ji", xmat) + xmat) # [8x8x3x3]
    eps_yderiv = 0.5*(np.einsum("...ji", ymat) + ymat) 
    eps_zderiv = 0.5*(np.einsum("...ji", zmat) + zmat)
    
    # (3.23)
    identity_ = np.array([np.identity(3) for _ in range(8)]) # [8x3x3]
    strains = 0.5*(np.einsum("...ji,...jk->...ik", matrices_, matrices_) - identity_) # [8x3x3]
    
    # (4.13)
    stresses = np.einsum("ijkl,...kl->...ij", C0, strains) # [8x3x3]
    gpos_state = np.zeros((8, 3))
    gpos_state[:, 0] += 0.125*np.einsum("...aji,...aij", eps_xderiv, stresses) # [8x8x3x3].[8x3x3] = [8]
    gpos_state[:, 1] += 0.125*np.einsum("...aji,...aij", eps_yderiv, stresses)
    gpos_state[:, 2] += 0.125*np.einsum("...aji,...aij", eps_zderiv, stresses)
    
    return h0_det*gpos_state
