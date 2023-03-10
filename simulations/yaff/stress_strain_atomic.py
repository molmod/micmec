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


import argparse

import pickle as pkl
import numpy as np
import numpy.linalg as la

from yaff import System, log
from micmec.analysis.tensor import voigt, voigt_inv
from micmec.utils import build_type

from molmod.units import pascal, angstrom

gigapascal = 1e9*pascal


def main(args):
    # Define the system and the force field.
    sys = System.from_file(args.chk_file)
    #ff = ff_lammps.load_ff(sys, args.pars_file, use_lammps=False)

    mass = np.sum(sys.masses)
    print(mass)

    # Add some noise to the initial Cartesian coordinates.
    max_noise = 0.05
    sys.pos += max_noise*(2.0*np.random.random_sample(sys.pos.shape) - 1.0)

    cell_init = sys.cell.rvecs.copy()

    # Perform an optimisation of the atomic positions and the cell parameters.
    ddof = StrainCellDOF(ff, gpos_rms=1e-6, dpos_rms=1e-4, grvecs_rms=1e-6, drvecs_rms=1e-4)
    osl = OptScreenLog(step=10)
    cgopt = CGOptimizer(ddof, hooks=[osl])
    cgopt.run()

    cell_eq = sys.cell.rvecs.copy()
    pos_eq = sys.pos.copy()

    max_dev = 0.001
    ndevs = 2
    one = np.identity(3)

    elasticity_matrix = np.zeros((6, 6))
    for idx in range(6):
        strain_devs = max_dev*np.linspace(-1.0, 1.0, ndevs)
        stress_devs = np.zeros((ndevs, 6))
        for idx_dev, strain_dev in enumerate(strain_devs):
            strain_vec = np.zeros(6)
            strain_vec[idx] = strain_dev
            strain = voigt_inv(strain_vec)
            cell = cell_eq @ la.sqrtm(2.0*strain + one)
            volume = la.det(cell)

            # Optimize coordinates.
            ff.update_rvecs(cell)
            ff.update_pos(pos_eq @ la.sqrtm(2.0*strain + one))
            cdof = CartesianDOF(ff, gpos_rms=1e-6, dpos_rms=1e-4)
            osl = OptScreenLog(step=50)
            cgopt = CGOptimizer(cdof)
            cgopt.run()

            stress_ = np.zeros((3, 3))
            ff.compute(vtens=stress_)
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
        log("- Equilibrium cell matrix [Å]:")
        log("    [[{:6.1f}, {:6.1f}, {:6.1f}],".format(*list(cell_eq[0]/angstrom)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}],".format(*list(cell_eq[1]/angstrom)))
        log("     [{:6.1f}, {:6.1f}, {:6.1f}]]".format(*list(cell_eq[2]/angstrom)))
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

    # Store the output as a new, micromechanical cell type.
    # Note that the convention for a nanocell matrix is different than the convention for a domain matrix, hence the transpose.
    output = build_type(
        material="UNKNOWN",
        mass=mass,
        cell0=cell_eq,
        elasticity0=elasticity_tensor,
        topology="mixed"
    )

    with open(args.pkl_file, "wb") as pklf:
        pkl.dump(output, pklf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Determine the elastic properties of an atomic system from finite deformations.")
    parser.add_argument("chk_file", type=str,
                        help=".chk filename of the input structure")
    parser.add_argument("pars_file", type=str,
                        help=".txt filename of the force field parameters")
    parser.add_argument("pkl_file", type=str,
                        help=".pickle filename of the output elastic properties")
    args = parser.parse_args()

    main(args)



