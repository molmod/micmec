#!/usr/bin/env python
# File name: nanocell_jax.py
# Description: The (correct) description of a nanocell in the micromechanical model, by means of the elastic deformation energy and its gradient, which is automatically derived with JAX.
# Author: Joachim Vandewalle
# Date: 10-06-2022

"""The (correct) description of a nanocell in the micromechanical model, 
by means of the elastic deformation energy and its gradient, which is automatically derived with JAX."""
# In the comments, we refer to equations in the master's thesis of Joachim Vandewalle.

# (JV) This implementation is very generally applicable, but it is slower than the default (`nanocell.py`).
# Its slowness can be due to:
#   1) a bad application of JAX's Just-In-Time compilation,
#   2) or the fact that I'm using the CPU version of CUDA, instead of the GPU version.
# I do not recommend using this script, at least not until someone properly compares its results to the results 
# of the default and finds a way to speed it up.

import numpy as np

import jax
import jax.numpy as jnp

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


# Just-In-Time compilation for computational speed-up.
@jax.jit
def elastic_energy(vertices_flat, h0, C0):
    vertices = vertices_flat.reshape((8, 3))

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


# Automatic differentiation.
def grad_(fun):
    return jax.grad(fun)
    


