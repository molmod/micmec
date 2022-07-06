#!/usr/bin/env python
# File name: advanced.py
# Description: Advanced analysis.
# Author: Joachim Vandewalle
# Date: 24-05-2022

"""Advanced trajectory analysis routines."""

import h5py as h5

import numpy as np

from molmod import boltzmann, pascal, angstrom, second, lightspeed, centimeter

from micmec.log import log
from micmec.analysis.tensor import voigt, voigt_inv
from micmec.analysis.utils import get_slice


__all__ = [
    "get_mass",
    "get_cell0",
    "get_elasticity0"
]

# The numbers refer to equations in the master's thesis of Joachim Vandewalle.
def _compute_volume(self, cell):
    """Compute the (instantaneous) volume of a domain.

    Parameters
    ----------
    cell : 
        SHAPE: (3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The (instantaneous) cell matrix.

    Returns
    -------
    volume : float
        The volume of the domain, in atomic units.
    
    """
    volume = np.linalg.det(cell) # (3.2)
    return volume


def _compute_strain(self, cell, cell0_inv):
    """Compute the strain tensor of a domain.
    
    Parameters
    ----------
    cell : 
        SHAPE: (3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The (instantaneous) cell matrix.
    cell0_inv : 
        SHAPE: (3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The inverse equilibrium cell matrix.

    Returns
    ------- 
    strain :   
        SHAPE: (3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The resulting strain tensor, in atomic units.
    
    Notes
    -----
    By Yaff convention, the domain vectors a, b and c appear as rows in every cell matrix.
    
    """
    strain = 0.5*((cell0_inv @ cell) @ (cell0_inv @ cell).T - np.identity(3)) # (3.3) 
    return strain 


def _compute_compliance(self, strain, strain0, volume0, temp0):
    """Compute the (instantaneous) compliance tensor of a domain.
    
    Parameters
    ----------
    strain : 
        SHAPE: (3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The (instantaneous) strain tensor.
    strain0 : 
        SHAPE: (3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The equilibrium strain tensor.
    volume0 : float
        The equilibrium volume of the domain.
    temp0 : float
        The equilibrium temperature of the domain.

    Returns
    ------- 
    compliance :   
        SHAPE: (3,3,3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The resulting compliance tensor, in atomic units.

    Notes
    -----
    In physics, there is no such thing as an instantaneous compliance tensor, but we use this terminology to
    clarify our calculation of the true compliance tensor, which is defined at (mechanical) equilibrium.
    The true compliance tensor is the mean of all instantaneous compliance tensors.
    
    """
    return (volume0/(boltzmann*kelvin*temp0))*np.tensordot(strain-strain0, strain-strain0, axes=0) # (3.19)


def get_mass(f):
    """Get the total mass of a domain from a .h5 file.
    
    Parameters
    ----------
    f : h5py.File object (open)
        A .h5 file, may be None if it is not available. 
    
    Returns
    -------
    mass : float
        The total mass of the domain, in atomic units.
    
    """
    masses = f["system/masses"]
    mass = np.sum(masses)
    return mass


def get_cell0(f, **kwargs):
    """Get the equilibrium cell matrix of a domain, based on trajectory data.
    
    Parameters
    ----------
    f : h5py.File object (open)
        A .h5 file containing the trajectory data.

    Notes
    -----
    The optional arguments of the `get_slice` function are also accepted in the form of keyword arguments.
    Please be aware that the simulation `domain` in this method can refer to:
        A) an atomistic force field simulation, in which case the trajectory of the domain is stored 
            in `trajectory/cell`;
        B) a micromechanical force field simulation, in which case the trajectory of the domain is stored
            in `trajectory/domain`.
    As such, this method can be used to analyze .h5 files from both Yaff and MicMec.
    
    """
    start, end, step = get_slice(f, **kwargs)
    # The "trajectory/cell" dataset is a time series of the cell matrix.
    if "trajectory/domain" in f:
        cells = f["trajectory/domain"][start:end:step]/log.length.conversion
    else:
        if "trajectory/cell" in f:
            cells = f["trajectory/cell"][start:end:step]/log.length.conversion
        else:
            raise IOError("File \"%s\" does not contain a Cell/Domain trajectory." % input_fn)
    # Be mindful of potential rotations of the simulation domain when computing the mean.
    cell0 = np.mean(cells, axis=0)
    return cell0


def get_elasticity0(f, **kwargs):
    """Get the elasticity tensor of a domain, based on trajectory data.
    
    Parameters
    ----------
    f : h5py.File object (open)
        A .h5 file containing the trajectory data.

    Notes
    -----
    The optional arguments of the `get_slice` function are also accepted in the form of keyword arguments.
    Please be aware that the simulation `domain` in this method can refer to:
        A) an atomistic force field simulation, in which case the trajectory of the domain is stored 
            in `trajectory/cell`;
        B) a micromechanical force field simulation, in which case the trajectory of the domain is stored
            in `trajectory/domain`.
    As such, this method can be used to analyze .h5 files from both Yaff and MicMec.
    
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
    temps = np.array(f["trajectory/temp"][start:end:step])
    temp0 = temps.mean()
    # Be mindful of potential rotations of the simulation domain when computing the mean.
    cell0 = np.mean(cells, axis=0)
    cell0_inv = np.linalg.inv(cell0)
    volume0 = _compute_volume(cell0)
    strains = np.array([_compute_strain(cell, cell0_inv) for cell in cells])
    strain0 = np.mean(strains, axis=0)
    # Calculating the equilibrium compliance tensor.
    compliances = np.array([_compute_compliance(strain, strain0, volume0, temp0) for strain in strains])
    compliance0 = np.mean(compliances, axis=0)       
    # Construct the compliance matrix (Voigt notation).
    compliance0_matrix = voigt(compliance0, mode="compliance")
    # Obtain the elasticity matrix by inversion.
    elasticity0_matrix = np.linalg.inv(compliance0_matrix)
    # Construct the elasticity tensor from the elasticity matrix.
    elasticity0 = voigt_inv(elasticity0_matrix, mode="elasticity")
    return elasticity0


