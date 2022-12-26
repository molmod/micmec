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


import numpy as np

import h5py
import json
import argparse

import matplotlib.pyplot as plt

from micmec.log import log
from micmec.system import System
from micmec.pes.mmff import MicMecForceField, ForcePartMechanical
from micmec.sampling.opt import OptScreenLog, QNOptimizer, CGOptimizer
from micmec.sampling.dof import CartesianDOF

from micmec.sampling.trajectory import HDF5Writer, XYZWriter

from molmod.units import kelvin, pascal, femtosecond, kjmol, angstrom

gigapascal = 1e9*pascal


def main(args):
    sys = System.from_file(args.input)
    sys.params["type1/effective_temp"] = 300.0
    orig_params = sys.params.copy()
    nstates = len(sys.params["type1/cell"])
    states = []
    for istate in range(nstates):
        states.append({
            "type1/cell": [orig_params["type1/cell"][istate]],
            "type1/elasticity": [orig_params["type1/elasticity"][istate]],
            "type1/free_energy": [orig_params["type1/free_energy"][istate]],
        })
    max_noise = 0.01
    fixed_nodes = []
    for istate, state in enumerate(states):
        log(f"Optimizing the nodal coordinates of STATE {istate}...")
        sys.params.update(state)
        sys.pos += max_noise*(2.0*np.random.random_sample(sys.pos.shape) - 1.0)
        fpm = ForcePartMechanical(sys)
        mmf = MicMecForceField(sys, [fpm])
        cdof = CartesianDOF(mmf)
            
        osl = OptScreenLog(step=1)
        
        qnopt = QNOptimizer(cdof, hooks=[osl])
        qnopt.run()
            
        fixed_nodes.append(np.concatenate([sys.pos[i] - sys.pos[0] for i in range(1, 8)]))
    num = 200
    sys.params.update(orig_params)
    intpol = np.array([np.linspace(low, high, num) for low, high in zip(fixed_nodes[0], fixed_nodes[1])])
    fixed = [0, 4]
    variable = [node for node in range(8) if node not in fixed]
    epot_profile = []
    for idx in range(num):
        sys.pos -= sys.pos[0]
        for i in range(1, 8):
            sys.pos[i] = intpol[3*(i-1):3*i, idx]
        #sys.pos[variable] += max_noise*(2.0*np.random.random_sample(sys.pos[variable].shape) - 1.0)
        fpm = ForcePartMechanical(sys)
        mmf = MicMecForceField(sys, [fpm])
#        cdof = CartesianDOF(mmf, select=variable)
#            
#        osl = OptScreenLog(step=1)
#        
#        qnopt = CGOptimizer(cdof, hooks=[osl])
#        qnopt.run()
        
        epot_profile.append(mmf.compute()/kjmol)
        
    plt.plot(epot_profile)
    plt.ylim(0.0, 50.0)
    plt.show()
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform a potential energy scan of a 1x1x1 bistable cell.")
    parser.add_argument("input", type=str, help=".chk filename of the input structure")
    parser.add_argument("-fn_png", type=str, default="bistable_scan.png",
                        help=".png filename of the output figure")

    args = parser.parse_args()
    main(args)


