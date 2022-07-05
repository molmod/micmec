#!/usr/bin/env python
# File name: system.py
# Description: The construction of a system of micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

import numpy as np

from micmec.utils import build_system

__all__ = ["build_output", "build_input"]

def build_output(data, colors_types, grid, pbc):
    """Build the user-defined output of the Builder Application (builder.py) and store it as a dictionary.
    
    Parameters
    ----------
    data : dict
        A dictionary with the names of the micromechanical cell types as keys. 
        The corresponding values are dictionaries which contain information about the cell type.
        Example: 
            data["fcu"] = {"elasticity": [np.array([[[[...]]]])], "cell": ...} 
            # the parameters of the cell type "fcu", including the elasticity tensor, equilibirum cell matrix etc.

    colors_types : dict
        A dictionary with integer keys. 
        These integers appear in the three-dimensional grid.
        The values corresponding to the keys are tuples of a color and the name of a type.
        Example: 
            colors_types[1] = ("#0000FF", "fcu") 
            # type 1 corresponds to the color blue and the name `fcu`

    grid : 
        SHAPE: (nx, ny, nz) 
        TYPE: numpy.ndarray
        DTYPE: int
        A three-dimensional grid that maps the types of cells present in the micromechanical system.
        An integer value of 0 in the grid signifies an empty cell, a vacancy.
        An integer value of 1 signifies a cell of type 1, etc.
        Example: 
            grid[kappa, lambda, mu] = 0 # empty cell at location (kappa, lambda, mu)
            grid[kappa_, lambda_, mu_] = 1 # cell of type 1 (`fcu`) at location (kappa_, lambda_, mu_)

    pbc : list of bools
        The domain vectors for which periodic boundary conditions should be enabled.
        Example:
            pbc = [True, True, True] 
            # periodic boundary conditions enabled for the (Cartesian) a, b and c domain vectors

    Returns
    -------
    output : dict
        A complete dictionary which is ready to be stored as a .chk file.
        It contains the information of the micromechanical system.

    Notes
    -----
    `build_input` is the inverse operation of `build_output`. 
    
    """
    new_data = {}
    for type_key, color_type in colors_types.items():
        color, name = color_type
        new_dict = {"color": color, "name": name}
        if name != "--NONE--":
            new_dict.update(data[name].copy())
        new_data[type_key] = new_dict

    return build_system(new_data, grid, pbc)


def build_input(output):
    """Build the input of the Builder Application (builder.py) from a complete dictionary.
    
    Parameters
    ----------
    output : dict
        A complete dictionary which is ready to be stored as a .chk file.
        It contains the information of a micromechanical system.

    Returns
    -------
    data : dict
        A dictionary with the names of the micromechanical cell types as keys. 
        The corresponding values are dictionaries which contain information about the cell type.
        Example: 
            data["fcu"] = {"elasticity": [np.array([[[[...]]]])], "cell": ...} 
            # the parameters of the cell type "fcu", including the elasticity tensor, equilibirum cell matrix etc.

    colors_types : dict
        A dictionary with integer keys. 
        These integers appear in the three-dimensional grid.
        The values corresponding to the keys are tuples of a color and the name of a type.
        Example: 
            colors_types[1] = ("#0000FF", "fcu") 
            # type 1 corresponds to the color blue and the name `fcu`

    grid : 
        SHAPE: (nx, ny, nz) 
        TYPE: numpy.ndarray
        DTYPE: int
        A three-dimensional grid that maps the types of cells present in the micromechanical system.
        An integer value of 0 in the grid signifies an empty cell, a vacancy.
        An integer value of 1 signifies a cell of type 1, etc.
        Example: 
            grid[kappa, lambda, mu] = 0 # empty cell at location (kappa, lambda, mu)
            grid[kappa_, lambda_, mu_] = 1 # cell of type 1 (`fcu`) at location (kappa_, lambda_, mu_)

    pbc : list of bools
        The domain vectors for which periodic boundary conditions should be enabled.
        Example:
            pbc = [True, True, True] 
            # periodic boundary conditions enabled for the (Cartesian) a, b and c domain vectors
    
    Notes
    -----
    `build_input` is the inverse operation of `build_output`. 
    
    """
    data = {}
    colors_types = {}
    grid = output["grid"]
    pbc = output["pbc"]

    temp = {}
    for field, value in output.items():
        if "type" in field and field != "types":
            field_lst = field.split("/")
            key = int(field_lst[0][4:])
            key_ = field_lst[1]
            if key not in temp.keys():
                temp[key] = []
            temp[key].append((key_, value))
    
    for key, tups in temp.items():
        type_dict = {}
        for tup in tups:
            if tup[0] == "name":
                name = tup[1]
            elif tup[0] == "color":
                color = tup[1]
            else:
                type_dict[tup[0]] = tup[1]
        if key != 0:
            data[name] = type_dict
        colors_types[key] = (color, name)
            
    return data, colors_types, grid, pbc


            



