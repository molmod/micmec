#!/usr/bin/env python

import numpy as np

import jax
import jax.numpy as jnp


from molmod.units import pascal


gigapascal = 1.0e9*pascal
V = {
    0: (0,0),
    1: (1,1), 
    2: (2,2), 
    3: (1,2),
    4: (0,2),
    5: (0,1)
}

def voigt_inv(matrix, mode=None):
    """Maps a 6x6 Voigt notation matrix into a 3x3x3x3 tensor."""
    
    tensor = np.zeros((3,3,3,3))
    
    if (mode is None) or (mode == "compliance"):
        for index, _ in np.ndenumerate(tensor):
            ij = tuple(sorted(index[0:2]))
            kl = tuple(sorted(index[2:4]))
            for key in V.keys():
                if V[key] == ij:
                    V_ij = key
                if V[key] == kl:
                    V_kl = key
            tensor[index] = matrix[(V_ij, V_kl)]
            if V_ij >= 3:
                tensor[index] *= 0.5
            if V_kl >= 3:
                tensor[index] *= 0.5
    
    elif (mode == "elasticity") or (mode == "stiffness"):
        for index, _ in np.ndenumerate(tensor):
            ij = tuple(sorted(index[0:2]))
            kl = tuple(sorted(index[2:4]))
            for key in V.keys():
                if V[key] == ij:
                    V_ij = key
                if V[key] == kl:
                    V_kl = key          
            tensor[index] = matrix[(V_ij, V_kl)]
    else:
        raise ValueError("Method voigt_inv() did not receive valid input for keyword 'mode'.")
    
    return tensor


h0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
h0_inv = np.linalg.inv(h0)

C0_mat = np.array([[50.0, 30.0, 30.0,  0.0,  0.0,  0.0],
                    [30.0, 50.0, 30.0,  0.0,  0.0,  0.0],
                    [30.0, 30.0, 50.0,  0.0,  0.0,  0.0],
                    [ 0.0,  0.0,  0.0, 10.0,  0.0,  0.0],
                    [ 0.0,  0.0,  0.0,  0.0, 10.0,  0.0],
                    [ 0.0,  0.0,  0.0,  0.0,  0.0, 10.0]])*gigapascal
C0 = voigt_inv(C0_mat, mode="elasticity")


pos0 = np.array([[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]])


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

elastic_energy_type0 = lambda verts: elastic_energy(verts, h0_inv, C0)
grad_type0 = jax.grad(elastic_energy_type0)
elastic_energy_type1 = lambda verts: elastic_energy(verts, 1.01*h0_inv, 1.1*C0)
grad_type1 = jax.grad(elastic_energy_type1)

for _ in range(100):
    pos = pos0 + 0.01*np.random.random((8, 3))
    print(elastic_energy_type0(pos.flatten()))
    print(elastic_energy_type1(pos.flatten()))
    print(grad_type0(pos.flatten()).reshape((8, 3)))
    print(grad_type1(pos.flatten()).reshape((8, 3)))
    print("\n")
    


