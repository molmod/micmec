#!/usr/bin/env python

import numpy as np

import jax
import jax.numpy as jnp


# Just-In-Time compilation for computational speed-up.
@jax.jit
def elastic_energy(vertices_flat, h0_inv, C0):
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

    energy_density = 0.5*jnp.einsum("...ij,ijkl,...kl", strains, C0, strains)
    energy = 0.125*jnp.einsum("i,i", jnp.linalg.det(matrices), energy_density)

    return energy


# Automatic differentiation.
def grad_elastic_energy(fun):
    return jax.grad(fun)
    


