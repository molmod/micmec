#!/usr/bin/env python
# File name: tensor.py
# Description: Auxiliary routines for tensors.
# Author: Joachim Vandewalle
# Date: 26-10-2021

"""Auxiliary routines for tensors."""

import numpy as np

from molmod.units import *
from molmod.constants import *

__all__ = [
    "print_3x3x3x3_tensor", 
    "print_6x6_matrix", 
    "voigt",
    "voigt_inv", 
    "plot_directional_young_modulus"
]

# The following routines improve the layout of tensors or matrices printed in a console.

def print_3x3x3x3_tensor(C):
    """Print out a (3 x 3 x 3 x 3) tensor as a square to allow the user a better overview."""
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
    return C_print_new


def print_6x6_matrix(C):
    """Print out a (6 x 6) matrix as a square to allow the user a better overview."""
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
    return C_print_new


V = {
    0: (0,0),
    1: (1,1), 
    2: (2,2), 
    3: (1,2),
    4: (0,2),
    5: (0,1)
}


def voigt(tensor, mode=None):
    """Maps a (3 x 3 x 3 x 3) tensor to a (6 x 6) Voigt notation matrix.
    
    Parameters
    ----------
    tensor : 
        SHAPE: (3,3,3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The tensor to be mapped to a Voigt notation matrix.
    mode : str, optional, default "compliance"
        Declare whether the input tensor is a compliance tensor (mode="compliance") or an elasticity tensor 
        (mode="elasticity" or mode="stiffness").

    Returns
    -------
    matrix :     
        SHAPE: (6,6) 
        TYPE: numpy.ndarray
        DTYPE: float
        The resulting Voigt notation matrix.

    Notes
    -----
    Voigt notation differs depending on whether the tensor is a compliance tensor or an elasticity tensor,
    hence the (optional) keyword `mode`.
        
    """
    
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
        raise IOError("Method `voigt_inv` did not receive valid input for keyword `mode`.") 

    return matrix


def voigt_inv(matrix, mode=None):
    """Maps a (6 x 6) Voigt notation matrix to a (3 x 3 x 3 x 3) tensor.
    
    Parameters
    ----------
    matrix : 
        SHAPE: (6,6) 
        TYPE: numpy.ndarray
        DTYPE: float
        The Voigt notation matrix to be mapped to a tensor.
    mode : str, optional, default "compliance"
        Declare whether the input matrix is a compliance matrix (mode="compliance") or an elasticity matrix 
        (mode="elasticity" or mode="stiffness").

    Returns
    -------
    tensor :     
        SHAPE: (3,3,3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        The resulting tensor.

    Notes
    -----
    Voigt notation differs depending on whether the tensor is a compliance tensor or an elasticity tensor,
    hence the (optional) keyword `mode`.
        
    """
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
        raise ValueError("Method `voigt_inv` did not receive valid input for keyword `mode`.")
    return tensor


def plot_directional_young_modulus(compliance_tensor, fn_png="directional_young_modulus.png"):
    """Plot the three-dimensional directional Young modulus, based on the compliance tensor.

    Parameters
    ----------
    compliance_tensor : 
        SHAPE: (3,3,3,3) 
        TYPE: numpy.ndarray
        DTYPE: float
        A fourth-order compliance tensor, expressed in atomic units.
    fn_png : str, optional
        The .png filename to write the figure to.
    
    """
    import matplotlib.pyplot as plt
    gigapascal = 1e9*pascal
    
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
        """Set three-dimensional plot axes to equal scale.

        Make the axes of a three-dimensional plot have equal scale so that spheres appear as spheres and cubes as cubes.  
        Required since `ax.axis("equal")` and `ax.set_aspect("equal")` don't work on three-dimensional plots.
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
        # Alter the limits manually if these automatic limits are not to your liking.
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])


    # Plot the surface.
    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")

    ax.set_box_aspect([1,1,1])
    ax.plot_surface(X, Y, Z)
    set_axes_equal(ax)
    ax.set_xlabel(r"$\mathrm{E_x \; [GPa]}$")
    ax.set_ylabel(r"$\mathrm{E_y \; [GPa]}$")
    ax.set_zlabel(r"$\mathrm{E_z \; [GPa]}$")
    
    ax = plt.gca()
    # Delete the tick labels of one or more axes.
    #ax.xaxis.set_ticklabels([])
    #ax.yaxis.set_ticklabels([])
    #ax.zaxis.set_ticklabels([])
    
    # Reduce the number of tick labels for better visibility.
    every_nth = 2
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    for n, label in enumerate(ax.zaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    # Delete the tick lines of one or more axes.
    #for line in ax.xaxis.get_ticklines():
    #    line.set_visible(False)
    #for line in ax.yaxis.get_ticklines():
    #    line.set_visible(False)
    #for line in ax.zaxis.get_ticklines():
    #    line.set_visible(False)
    
    plt.show()


