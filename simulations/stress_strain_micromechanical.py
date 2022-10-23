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
import pickle as pkl

import h5py
import argparse

import scipy.linalg as la

from micmec.system import System
from micmec.pes.mmff import MicMecForceField, ForcePartMechanical
from micmec.sampling.opt import OptScreenLog, QNOptimizer
from micmec.sampling.dof import CartesianDOF, StrainCellDOF

from micmec.analysis.tensor import voigt, voigt_inv
from micmec.utils import build_type

from molmod.units import kelvin, pascal, femtosecond, angstrom
from micmec.log import log

gigapascal = 1e9*pascal

def main(args):
    # Define the system and the force field.
    sys = System.from_file(args.chk_file)
    mmf = MicMecForceField(sys, [ForcePartMechanical(sys)])

    mass = np.sum(sys.masses)

    # Add some noise to the initial Cartesian coordinates.
    max_noise = 0.10
    sys.pos += max_noise*(2.0*np.random.random_sample(sys.pos.shape) - 1.0)

    domain_init = sys.domain.rvecs.copy()

    # Perform an optimisation of the atomic positions and the domain parameters.
    ddof = StrainCellDOF(mmf, gpos_rms=1e-8, dpos_rms=1e-6, grvecs_rms=1e-8, drvecs_rms=1e-6)
    osl = OptScreenLog(step=1)
    qnopt = QNOptimizer(ddof, hooks=[osl])
    qnopt.run()

    domain_eq = sys.domain.rvecs.copy()
    pos_eq = sys.pos.copy()

    max_dev = 0.003
    ndevs = 4
    one = np.identity(3)

    elasticity_matrix = np.zeros((6, 6))
    for idx in range(6):
        strain_devs = max_dev*np.linspace(-1.0, 1.0, ndevs)
        stress_devs = np.zeros((ndevs, 6))
        for idx_dev, strain_dev in enumerate(strain_devs):
            strain_vec = np.zeros(6)
            strain_vec[idx] = strain_dev
            strain = voigt_inv(strain_vec)
            domain = domain_eq @ la.sqrtm(2.0*strain + one)
            volume = la.det(domain)
            
            # Optimize coordinates.
            mmf.update_rvecs(domain)
            mmf.update_pos(pos_eq + max_noise*(2.0*np.random.random_sample(pos_eq.shape) - 1.0))
            cdof = CartesianDOF(mmf, gpos_rms=1e-7, dpos_rms=1e-5)
            osl = OptScreenLog(step=1)
            qnopt = QNOptimizer(cdof, hooks=[osl])
            qnopt.run()
            
            stress_ = np.zeros((3, 3))
            mmf.compute(vtens=stress_)
            stress = stress_/volume
            stress_vec = voigt(stress, mode="stress")
            stress_devs[idx_dev, :] = stress_vec
            with log.section("ITER"):
                log("Intermediate results.")
                log.hline()
                log(" ")
                log("- Strain vector [-]:")
                log("     [{:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}, {:6.3f}],".format(*list(strain_vec)))
                log("- Stress vector [GPa]:")
                log("     [{:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}, {:6.2f}],".format(*list(stress_vec/gigapascal)))
                log(" ")
                log.hline()
        
        elasticity_vec = np.polyfit(strain_devs, stress_devs, deg=1)[0, :]
        elasticity_matrix[:, idx] = elasticity_vec

    with log.section("POST"):
        log("Final results.")
        log.hline()
        log(" ")
        log("- Equilibrium domain matrix [Å]:")
        log("    [[{:6.1f}, {:6.1f}, {:6.1f}],".format(*list(domain_eq[0]/angstrom)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}],".format(*list(domain_eq[1]/angstrom)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}]]".format(*list(domain_eq[2]/angstrom)))
        log("- Elasticity matrix [GPa]:")
        log("    [[{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(elasticity_matrix[0]/gigapascal)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(elasticity_matrix[1]/gigapascal)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(elasticity_matrix[2]/gigapascal)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(elasticity_matrix[3]/gigapascal)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}],".format(*list(elasticity_matrix[4]/gigapascal)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}, {:6.1f}]]".format(*list(elasticity_matrix[5]/gigapascal)))
        log(" ")
        log.hline()

    elasticity_tensor = voigt_inv(elasticity_matrix, mode="elasticity")

    # Store the output as a new, micromechanical domain type.
    # Note that the convention for a nanodomain matrix is different than the convention for a domain matrix, hence the transpose.
    output = build_type(
        material="UiO-66(Zr)", 
        mass=mass, 
        cell0=domain_eq,
        elasticity0=elasticity_tensor,
        topology="configuration"
    )

    with open(args.pkl_file, "wb") as pklf:
        pkl.dump(output, pklf)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Determine the elastic properties of an atomic system from finite deformations.")
    parser.add_argument(
        "chk_file", 
        type=str,
        help=".chk filename of the input structure"
    )
    parser.add_argument(
        "pkl_file", 
        type=str,
        help=".pickle filename of the output elastic properties"
    )
    args = parser.parse_args()
    
    main(args)




