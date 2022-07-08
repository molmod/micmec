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


"""Generate test types for further use in a micromechanical system."""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from micmec.analysis.tensor import voigt, voigt_inv
from micmec.utils import build_type

from molmod.units import kelvin, pascal, angstrom

gigapascal = 1e9*pascal

def main():
    # Define two toy systems with very easy parameters in atomic units.
    mass = 10000000
    unit_cell_length = 10.0*angstrom
    cell = np.identity(3)*unit_cell_length
    cell_shrunk = 0.5*cell
    
    elasticity_matrix = np.array([[50.0, 30.0, 30.0,  0.0,  0.0,  0.0],
                                  [30.0, 50.0, 30.0,  0.0,  0.0,  0.0],
                                  [30.0, 30.0, 50.0,  0.0,  0.0,  0.0],
                                  [ 0.0,  0.0,  0.0, 10.0,  0.0,  0.0],
                                  [ 0.0,  0.0,  0.0,  0.0, 10.0,  0.0],
                                  [ 0.0,  0.0,  0.0,  0.0,  0.0, 10.0]])*gigapascal
    elasticity = voigt_inv(elasticity_matrix, mode="elasticity")
    
    type_test = build_type(material="test", 
                            mass=mass, 
                            cell0=cell,
                            elasticity0=elasticity,
                            topology="test")
    type_shrunk = build_type(material="test_shrunk", 
                            mass=mass, 
                            cell0=cell_shrunk,
                            elasticity0=elasticity,
                            topology="test")
    
    with open("type_test.pickle", "wb") as f:
        pkl.dump(type_test, f)
    
    with open("type_shrunk.pickle", "wb") as f:
        pkl.dump(type_shrunk, f)

if __name__ == '__main__':
    main()


