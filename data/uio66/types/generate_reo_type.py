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


"""Generate the **reo** type for further use in a micromechanical system."""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import h5py

from micmec.analysis.tensor import voigt, voigt_inv
from micmec.analysis.advanced import get_mass, get_cell0, get_elasticity0
from micmec.utils import build_type

from molmod.units import kelvin, pascal, angstrom

gigapascal = 1e9*pascal

def main():
    
    f = h5py.File("../atomic/md/output_reo_0MPa.h5", "r")
    mass = get_mass(f)
    cell = get_cell0(f, start=2000)
    elasticity = get_elasticity0(f, start=2000)
    f.close()

    type_reo = build_type(material="UiO-66(Zr)", 
                            mass=mass, 
                            cell0=cell,
                            elasticity0=elasticity,
                            topology="reo")

    with open("type_reo_new.pickle", "wb") as f:
        pkl.dump(type_reo, f)

if __name__ == '__main__':
    main()


