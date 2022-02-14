#!/usr/bin/env python
# File name: tensor.py
# Description:
# Author: Joachim Vandewalle
# Date: 26-10-2021

import numpy as np
import matplotlib.pyplot as plt

from molmod.units import *
from molmod.constants import *

__all__ = ["print_tensor", "print_matrix", "voigt", "voigt_inv", "plot_directional_young_modulus"]

def print_tensor(C):
    """Print out a fourth-order 3x3x3x3 tensor as a square to allow the user a better overview."""
    C_print = str(C)
    C_print_new = ""
    C_print_lst = C_print.split("\n")
    C_print_lst_new = [c for c in C_print_lst if c != ""]
    for i, c in enumerate(C_print_lst_new):
        C_print_new += c
        if (i+1)%3 == 0 and i != 0:
            C_print_new += "\n"
        else:
            pass
    print(C_print_new)
    return None


def print_matrix(C):
    """Print out a 6x6 matrix as a square to allow the user a better overview."""
    C_print = str(C)
    C_print_new = ""
    C_print_lst = C_print.split("\n")
    C_print_lst_new = [c for c in C_print_lst if c != ""]
    for i, c in enumerate(C_print_lst_new):
        C_print_new += c
        if c[-1] == "]": 
            C_print_new += "\n"
        else: 
            pass
    print(C_print_new)
    return None


V = {
    0: (0,0),
    1: (1,1), 
    2: (2,2), 
    3: (1,2),
    4: (0,2),
    5: (0,1)
}


def voigt(tensor, mode=None):
    """Maps a 3x3x3x3 tensor into a 6x6 voigt notation matrix."""
    
    matrix = np.zeros((6,6))
    
    if (mode is None) or (mode == "compliance"):
        for index, _ in np.ndenumerate(matrix):
            matrix[index] = tensor[V[index[0]] + V[index[1]]]
            if index[0] >= 3:
                matrix[index] *= 2.0
            if index[1] >= 3:
                matrix[index] *= 2.0
    
    elif (mode == "elasticity") or (mode == "stiffness"):
        for index, _ in np.ndenumerate(matrix):
            matrix[index] = tensor[V[index[0]] + V[index[1]]]
    else:
        raise ValueError("Method voigt() did not receive valid input for keyword 'mode'.") 

    return matrix


def voigt_inv(matrix, mode=None):
    """Maps a 6x6 voigt notation matrix into a 3x3x3x3 tensor."""
    
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


def plot_directional_young_modulus(compliance_tensor):
    """Plot the 3D directional Young modulus, based on the compliance tensor."""
    
    # Create the mesh in spherical coordinates and compute corresponding E.
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    PHI, THETA = np.meshgrid(phi, theta)
    U = [np.cos(PHI)*np.sin(THETA), np.sin(PHI)*np.sin(THETA), np.cos(THETA)]

    E = np.zeros(THETA.shape)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    E += U[i]*U[j]*U[k]*U[l]*compliance_tensor[i,j,k,l]
    E = np.absolute(1/E)/gigapascal

    # Express the mesh in the cartesian system.
    X, Y, Z = E*np.cos(PHI)*np.sin(THETA), E*np.sin(PHI)*np.sin(THETA), E*np.cos(THETA)


    def set_axes_equal(ax: plt.Axes):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5*np.max(np.abs(limits[:,1] - limits[:,0]))
        _set_axes_radius(ax, origin, radius)

    def _set_axes_radius(ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])


    # Plot the surface.
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')

    ax.set_box_aspect([1,1,1])
    ax.plot_surface(X, Y, Z)
    set_axes_equal(ax)
    ax.set_xlabel(r"$\mathrm{E_x \; [GPa]}$")
    ax.set_ylabel(r"$\mathrm{E_y \; [GPa]}$")
    ax.set_zlabel(r"$\mathrm{E_z \; [GPa]}$")
    plt.show()


