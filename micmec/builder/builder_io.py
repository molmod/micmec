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


"""Input and output handling of the Micromechanical Model Builder."""


from micmec.utils import build_system


__all__ = ["build_output", "build_input"]


def build_output(data, colors_types, grid, pbc):
    """Build the output of the Builder application (``builder.py``) and store it as a dictionary.

    Parameters
    ----------
    data : dict
        A dictionary with the names of the micromechanical cell types as keys.
        The corresponding values are dictionaries which contain information about the cell types.
    colors_types : dict
        A dictionary with integer keys.
        These integers appear in the three-dimensional grid.
        The values corresponding to the keys are tuples of a color and the name of a type.
    grid : numpy.ndarray, dtype=int, shape=(``nx``, ``ny``, ``nz``)
        A three-dimensional grid that maps the types of cells present in the micromechanical system.
        An integer value of 0 in the grid signifies an empty cell, a vacancy.
        An integer value of 1 signifies a cell of type 1, a value of 2 signifies a cell of type 2, etc.
    pbc : list of bool
        The domain vectors for which periodic boundary conditions should be enabled.

    Returns
    -------
    output : dict
        A dictionary which is ready to be stored as a CHK file, containing a complete description of the micromechanical system.

    Notes
    -----
    ``build_input`` is the inverse operation of ``build_output``.
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
    """Build the input of the Builder application (``builder.py``) from a dictionary.

    Parameters
    ----------
    output : dict
        A dictionary which is ready to be stored as a CHK file, containing a complete description of the micromechanical system.

    Returns
    -------
    data : dict
        A dictionary with the names of the micromechanical cell types as keys.
        The corresponding values are dictionaries which contain information about the cell types.
    colors_types : dict
        A dictionary with integer keys.
        These integers appear in the three-dimensional grid.
        The values corresponding to the keys are tuples of a color and the name of a type.
    grid : numpy.ndarray, dtype=int, shape=(``nx``, ``ny``, ``nz``)
        A three-dimensional grid that maps the types of cells present in the micromechanical system.
        An integer value of 0 in the grid signifies an empty cell, a vacancy.
        An integer value of 1 signifies a cell of type 1, a value of 2 signifies a cell of type 2, etc.
    pbc : list of bool
        The domain vectors for which periodic boundary conditions should be enabled.

    Notes
    -----
    ``build_input`` is the inverse operation of ``build_output``.
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
