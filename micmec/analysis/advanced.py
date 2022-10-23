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


"""Advanced trajectory analysis routines."""

import h5py as h5

import numpy as np

from molmod import boltzmann, pascal, angstrom, second, lightspeed, centimeter, kelvin

from micmec.log import log
from micmec.analysis.tensor import voigt, voigt_inv
from micmec.analysis.utils import get_slice


__all__ = [
    "get_mass",
    "get_cell0",
    "get_elasticity0"
]


# The numbers refer to equations in the master's thesis of Joachim Vandewalle.
def _compute_volume(cell):
    """Compute the (instantaneous) volume of a simulation domain.

    Parameters
    ----------
    cell : numpy.ndarray, shape=(3, 3)
        The (instantaneous) cell matrix.

    Returns
    -------
    volume : float
        The volume of the domain, in atomic units.
    """
    volume = np.linalg.det(cell) # (3.2)
    return volume


def _compute_strain(cell, cell0_inv):
    """Compute the strain tensor of a simulation domain.
    
    Parameters
    ----------
    cell : numpy.ndarray, shape=(3, 3)
        The (instantaneous) cell matrix.
    cell0_inv : numpy.ndarray, shape=(3, 3)
        The inverse equilibrium cell matrix.

    Returns
    ------- 
    strain : numpy.ndarray, shape=(3, 3)
        The resulting strain tensor, in atomic units.
    
    Notes
    -----
    By Yaff convention, the domain vectors ``a``, ``b`` and ``c`` appear as rows in every cell matrix.
    """
    strain = 0.5*((cell0_inv @ cell) @ (cell0_inv @ cell).T - np.identity(3)) # (3.3) 
    return strain 


def _compute_compliance(strain, strain0, volume0, temp0):
    """Compute the (instantaneous) compliance tensor of a simulation domain.
    
    Parameters
    ----------
    strain : numpy.ndarray, shape=(3, 3)
        The (instantaneous) strain tensor.
    strain0 : numpy.ndarray, shape=(3, 3)
        The equilibrium strain tensor.
    volume0 : float
        The equilibrium volume of the domain.
    temp0 : float
        The equilibrium temperature of the domain.

    Returns
    ------- 
    compliance : numpy.ndarray, shape=(3, 3, 3, 3)
        The resulting compliance tensor, in atomic units.

    Notes
    -----
    In physics, there is no such thing as an instantaneous compliance tensor, but we use this terminology to
    clarify our calculation of the true compliance tensor, which is defined at (mechanical) equilibrium.
    The true compliance tensor is the mean of all instantaneous compliance tensors, according to the ergodic hypothesis.
    """
    return (volume0/(boltzmann*kelvin*temp0))*np.tensordot(strain-strain0, strain-strain0, axes=0) # (3.19)


def get_mass(f):
    """Get the total mass of a simulation domain from an HDF5 file.
    
    Parameters
    ----------
    f : h5py.File
        An HDF5 file, may be ``None`` if it is not available. 
    
    Returns
    -------
    mass : float
        The total mass of the domain, in atomic units.
    """
    masses = f["system/masses"]
    mass = np.sum(masses)
    return mass


def get_cell0(f, **kwargs):
    """Get the equilibrium cell matrix of a simulation domain, based on trajectory data.
    
    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.
    
    Returns
    ------- 
    cell0 : numpy.ndarray, shape=(3, 3)
        The equilibrium cell matrix, in atomic units.

    Notes
    -----
    The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.
    Please be aware that the simulation domain in this method can refer to:
    
    -   an atomistic force field simulation, in which case the trajectory of the domain is stored in ``trajectory/cell``;
    -   a micromechanical force field simulation, in which case the trajectory of the domain is stored in ``trajectory/domain``.
    
    As such, this method can be used to analyze HDF5 files from both Yaff and MicMec.
    """
    start, end, step = get_slice(f, **kwargs)
    # The "trajectory/cell" dataset is a time series of the cell matrix.
    if "trajectory/domain" in f:
        cells = f["trajectory/domain"][start:end:step]
    else:
        if "trajectory/cell" in f:
            cells = f["trajectory/cell"][start:end:step]
        else:
            raise IOError("File \"%s\" does not contain a Cell/Domain trajectory." % input_fn)
    # Be mindful of potential rotations of the simulation domain when computing the mean.
    cell0 = np.mean(cells, axis=0)
    return cell0


def get_elasticity0(f, **kwargs):
    """Get the elasticity tensor of a simulation domain, based on trajectory data.
    
    Parameters
    ----------
    f : h5py.File
        An HDF5 file containing the trajectory data.

    Returns
    ------- 
    elasticity0 : numpy.ndarray, shape=(3, 3, 3, 3)
        The elasticity tensor, in atomic units.

    Notes
    -----
    The optional arguments of the ``get_slice`` function are also accepted in the form of keyword arguments.
    Please be aware that the simulation domain in this method can refer to:
    
    -   an atomistic force field simulation, in which case the trajectory of the domain is stored in ``trajectory/cell``;
    -   a micromechanical force field simulation, in which case the trajectory of the domain is stored in ``trajectory/domain``.
    
    As such, this method can be used to analyze HDF5 files from both Yaff and MicMec.
    """
    start, end, step = get_slice(f, **kwargs)
    # The "trajectory/cell" dataset is a time series of the cell matrix.
    if "trajectory/domain" in f:
        cells = np.array(f["trajectory/domain"][start:end:step])
    else:
        if "trajectory/cell" in f:
            cells = np.array(f["trajectory/cell"][start:end:step])
        else:
            raise IOError("File \"%s\" does not contain a domain trajectory." % input_fn)
    nsamples = len(cells)
    temp0 = np.mean(f["trajectory/temp"][start:end:step])
    # Be mindful of potential rotations of the simulation domain when computing the mean.
    cell0 = np.mean(cells, axis=0)
    cell0_inv = np.linalg.inv(cell0)
    volume0 = _compute_volume(cell0)
    strain0 = np.zeros((3, 3))
    for cell in cells:
        strain0 += _compute_strain(cell, cell0_inv)
    strain0 /= nsamples
    # Calculating the equilibrium compliance tensor.
    compliance0 = np.zeros((3, 3, 3, 3))
    for cell in cells:
        strain = _compute_strain(cell, cell0_inv)
        compliance0 += _compute_compliance(strain, strain0, volume0, temp0)
    compliance0 /= nsamples
    # Construct the compliance matrix (Voigt notation).
    compliance0_matrix = voigt(compliance0, mode="compliance")
    # Obtain the elasticity matrix by inversion.
    elasticity0_matrix = np.linalg.inv(compliance0_matrix)
    # Construct the elasticity tensor from the elasticity matrix.
    elasticity0 = voigt_inv(elasticity0_matrix, mode="elasticity")
    return elasticity0

