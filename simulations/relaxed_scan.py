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
from micmec.sampling.opt import OptScreenLog, QNOptimizer
from micmec.sampling.dof import CartesianDOF

from micmec.sampling.trajectory import HDF5Writer, XYZWriter

from molmod.units import kelvin, pascal, femtosecond, kjmol, angstrom

gigapascal = 1e9*pascal

def main(input_fns, fn_png, lower_limit, upper_limit):
    all_scalings = []
    all_energy_densities = []
    
    for idx, input_fn in enumerate(input_fns):
        scalings = np.linspace(lower_limit, upper_limit, 20)
        epots = []
        volumes = []
        for scaling in scalings:
            sys = System.from_file(input_fn)
            orig_rvecs = sys.domain.rvecs.copy()
            sys.domain.update_rvecs(orig_rvecs*scaling)
            fpm = ForcePartMechanical(sys)
            mmf = MicMecForceField(sys, [fpm])
            cdof = CartesianDOF(mmf)
            
            osl = OptScreenLog(step=1)
            
            qnopt = QNOptimizer(cdof, hooks=[osl])
            qnopt.run()
            
            epots.append(qnopt.epot)
            volumes.append(np.linalg.det(sys.domain.rvecs)) 
            
        V = np.array(volumes)
        E = np.array(epots)
        dV = np.diff(volumes)
        dV2 = (0.5*(dV[1:] + dV[:-1]))**2
        d2E = E[:-2] - 2.0*E[1:-1] + E[2:]
        # Assume that the original rvecs are the equilibrium rvecs of the simulation domain.
        bulk_modulus = np.mean(np.linalg.det(orig_rvecs)*d2E/dV2)
        
        if log.do_medium:
            with log.section("SCAN"):
                log.hline()
                s1 = "A relaxed potential energy scan has been performed by varying the volume of the simulation domain, isotropically, "
                s2 = f"from {(100*lower_limit):.0f} % to {(100*upper_limit):.0f} % of its original volume. "
                s3 = f"The result has been saved as `{fn_png}`. "
                s4 = f"Additionally, a static bulk modulus of {(bulk_modulus/gigapascal):.2f} GPa has been calculated for this simulation domain."
                log(s1+s2+s3+s4)
                log.hline()
        
        if idx > 0:
            plt.plot(scalings, np.array(epots)/np.array(volumes)/(kjmol/angstrom**3), "--")
        else:
            plt.plot(scalings, np.array(epots)/np.array(volumes)/(kjmol/angstrom**3))
        all_scalings.append(scalings.tolist())
        all_energy_densities.append((np.array(epots)/np.array(volumes)/(kjmol/angstrom**3)).tolist())
            
    #plt.plot(scalings, 0.5**(13*gigapascal/np.linalg.det(orig_rvecs))*(scalings - 1.0)**2)
    plt.xlabel(r"$V/V_0$ [-]")
    plt.ylabel("POTENTIAL ENERGY DENSITY [kJ/mol/Å³]")
    plt.xlim(scalings[0], scalings[-1])
    plt.ylim(0.0)
    plt.grid()
    plt.savefig(fn_png)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform a relaxed potential energy scan.")
    parser.add_argument("input_fn", nargs="+",
                        help=".chk filenames of the input structures")
    parser.add_argument("-fn_png", type=str, default="relaxed_scan.png",
                        help=".png filename of the output figure")
    parser.add_argument("-lower_limit", type=float, default=0.98,
                        help="lower limit of the volume scaling")
    parser.add_argument("-upper_limit", type=float, default=1.02,
                        help="upper limit of the volume scaling")

    args = parser.parse_args()
    main(args.input_fn, args.fn_png, args.lower_limit, args.upper_limit)


