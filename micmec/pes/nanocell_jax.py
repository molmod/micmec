#!/usr/bin/env python
# File name: nanocell_jax.py
# Description: The (correct) description of a nanocell in the micromechanical model, using JAX.
# Author: Joachim Vandewalle
# Date: 10-06-2022

"""The (correct) description of a nanocell in the micromechanical model, using JAX.

This implementation is very generally applicable, but it is slower than the default (`nanocell.py`).
Its slowness is likely due to the computational overhead of jax.numpy.
It is not recommended to use this script, at least not until someone properly compares its results to the results 
of the default.
"""

# In the comments, we refer to equations in the master's thesis of Joachim Vandewalle.

import numpy as np
import jax.numpy as jnp

__all__ = ["elastic_energy"]

# Construct a multiplicator array.
# This array converts the eight Cartesian coordinate vectors of a cell's surrounding nodes into eight matrix representations.
multiplicator = jnp.array([
    [[-1, 1, 0, 0, 0, 0, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], [-1, 0, 0, 1, 0, 0, 0, 0]],
    [[-1, 1, 0, 0, 0, 0, 0, 0], [ 0,-1, 0, 0, 1, 0, 0, 0], [ 0,-1, 0, 0, 0, 1, 0, 0]],
    [[ 0, 0,-1, 0, 1, 0, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], [ 0, 0,-1, 0, 0, 0, 1, 0]],
    [[ 0, 0, 0,-1, 0, 1, 0, 0], [ 0, 0, 0,-1, 0, 0, 1, 0], [-1, 0, 0, 1, 0, 0, 0, 0]],
    [[ 0, 0,-1, 0, 1, 0, 0, 0], [ 0,-1, 0, 0, 1, 0, 0, 0], [ 0, 0, 0, 0,-1, 0, 0, 1]],
    [[ 0, 0, 0,-1, 0, 1, 0, 0], [ 0, 0, 0, 0, 0,-1, 0, 1], [ 0,-1, 0, 0, 0, 1, 0, 0]],
    [[ 0, 0, 0, 0, 0, 0,-1, 1], [ 0, 0, 0,-1, 0, 0, 1, 0], [ 0, 0,-1, 0, 0, 0, 1, 0]],
    [[ 0, 0, 0, 0, 0, 0,-1, 1], [ 0, 0, 0, 0, 0,-1, 0, 1], [ 0, 0, 0, 0,-1, 0, 0, 1]]
])


def elastic_energy(vertices_flat, h0, C0):
    """The elastic energy of a nanocell, with respect to one of its metastable states with parameters h0 and C0.
        
    Parameters
    ----------
    vertices_flat : 
        SHAPE: (24,) 
        TYPE: numpy.ndarray
        DTYPE: float
        The (Cartesian) coordinates of the surrounding nodes (i.e. the vertices).
    h0 :
        SHAPE: (3, 3) 
        TYPE: numpy.ndarray
        DTYPE: float    
        The equilibrium cell matrix.
    C0 :
        SHAPE: (3, 3, 3, 3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The elasticity tensor.

    Returns
    -------
    float
        The elastic energy.
    
    Notes
    -----
    At first sight, the equations for bistable nanocells might seem absent from this derivation.
    They are absent here, but they have been implemented in the `mmff.py` script.
    This elastic energy is only the energy of a single metastable state of a nanocell.
    
    """
    vertices = vertices_flat.reshape((8, 3))

    # The equations below are the same as in the default (`nanocell.py`).
    # Feel free to plug in a different energy expression; its gradient will be computed automatically.
    # You do, however, need to use jax.numpy, which has some limitations and some computational overhead.
    
    # (3.20)
    matrices = jnp.einsum("...i,ij->...j", multiplicator, vertices)
    
    # (3.23)
    matrices_ = jnp.einsum("...ji,kj->...ik", matrices, jnp.linalg.inv(h0))
    identity_ = jnp.array([jnp.identity(3) for _ in range(8)])
    strains = 0.5*(jnp.einsum("...ji,...jk->...ik", matrices_, matrices_) - identity_)

    # (4.11)
    energy_density = 0.5*jnp.einsum("...ij,ijkl,...kl", strains, C0, strains)
    energy = 0.125*jnp.einsum("i->", energy_density)*jnp.linalg.det(h0)
    
    return energy

    


