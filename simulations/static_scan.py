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

import json
import argparse

import matplotlib.pyplot as plt

from molmod.units import angstrom, pascal, femtosecond, kjmol

from micmec.log import log
from micmec.system import System
from micmec.pes.mmff import MicMecForceField, ForcePartMechanical

gigapascal = (1e9)*pascal


def main(input_fn, fn_png, node_idx, num):
    
    # Initialize the system and the micromechanical force field.
    sys = System.from_file(input_fn)
    fpm = ForcePartMechanical(sys)
    mmf = MicMecForceField(sys, [fpm])

    pos = sys.pos.copy()
    pos_ref = sys.pos.copy()

    # It is preferred that the scan has an even number of scan locations (num) in each dimension.
    # After taking the numerical derivative, then it will be possible to obtain values exactly in the center
    # of the scanning range if num is even.
    mid = (num - 2)//2

    # Define a maximum deviation of the node in each dimension, in atomic units.
    # This should not exceed the dimensions of the cell.
    unit_cell_length = np.sqrt(np.sum((sys.pos[1] - sys.pos[0])**2))
    max_dev = 0.01*unit_cell_length

    # Scanning ranges.
    x_range = max_dev*np.linspace(-1, 1, num)
    y_range = max_dev*np.linspace(-1, 1, num)
    z_range = max_dev*np.linspace(-1, 1, num)

    # The scan will evaluate the ENERGY_POT and the forces at each scan location (x, y, z).
    X_ana, Y_ana, Z_ana = np.meshgrid(x_range, y_range, z_range, indexing="ij")

    ENERGY_POT = np.zeros((num, num, num))
    ZERO = np.zeros((num, num, num))
    FX_ana = np.zeros((num, num, num))
    FY_ana = np.zeros((num, num, num))

    mass = sys.masses[node_idx]

    for SCAN_INDEX, _ in np.ndenumerate(ZERO):
        gpos = np.zeros(pos.shape)
        # Manually move one node to the current scanning location.
        pos[node_idx] = pos_ref[node_idx] + np.array([X_ana[SCAN_INDEX], Y_ana[SCAN_INDEX], Z_ana[SCAN_INDEX]])
        mmf.update_pos(pos)
        ENERGY_POT[SCAN_INDEX] = mmf.compute(gpos)
        FX_ana[SCAN_INDEX] = -gpos[node_idx, 0]
        FY_ana[SCAN_INDEX] = -gpos[node_idx, 1]

    FX_num = -np.diff(ENERGY_POT, axis=0)/np.diff(X_ana, axis=0)
    X_num = 0.5*(X_ana[:-1, :, :] + X_ana[1:, :, :])

    rico = np.polyfit(X_num[:, mid, mid], FX_num[:, mid, mid], 1)[0]
    force_con = -rico
    timestep = np.pi*np.sqrt(mass/force_con)
    if log.do_medium:
        with log.section("SCAN"):
            log.hline()
            s1 = "A static potential energy scan has been performed " \
                 f"by varying the coordinates of a single node (node {node_idx}). "
            s2 = f"The result has been saved as `{fn_png}`. "
            s3 = "Additionally, Hooke's law has been verfied. "
            s4 = "From the force constant of Hooke's law, a maximal " \
                 f"timestep of {(timestep/femtosecond):.2f} fs is estimated for micromechanical MD simulations."
            log(s1+s2+s3+s4)
            log.hline()

    fig = plt.figure(figsize=(13, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(X_ana[:, mid, mid]/angstrom, FX_ana[:, mid, mid]*angstrom/kjmol,
             "o:", mfc="none", ms=16, label="analytical expression", color="blue")
    ax1.plot(X_num[:, mid, mid]/angstrom, FX_num[:, mid, mid]*angstrom/kjmol,
             "x", ms=16, label="numerical derivative", color="red")
    ax1.set_xlabel("$x - x_0$ [Å]")
    ax1.set_ylabel("$f_x$ [kJ/mol/Å]")
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax2.plot_surface(X_ana[:, :, mid]/angstrom, Y_ana[:, :, mid]/angstrom, ENERGY_POT[:, :, mid]/kjmol, cmap="viridis")
    ax2.set_xlabel("$x - x_0$ [Å]")
    ax2.set_ylabel("$y - y_0$ [Å]")
    ax2.set_zlabel("POTENTIAL ENERGY [kJ/mol]")

    with open("output_energy.json", "w") as jfile:
        jobj = {
            "X_ana": (X_ana[:, :, mid]/angstrom).tolist(),
            "Y_ana": (Y_ana[:, :, mid]/angstrom).tolist(),
            "ENERGY_POT": (ENERGY_POT[:, :, mid]/kjmol).tolist(),
        }
        json.dump(jobj, jfile)

    with open("output_forces.json", "w") as jfile:
        jobj = {
            "X_ana": (X_ana[:, mid, mid]/angstrom).tolist(),
            "X_num": (X_num[:, mid, mid]/angstrom).tolist(),
            "FX_ana": (FX_ana[:, mid, mid]*angstrom/kjmol).tolist(),
            "FX_num": (FX_num[:, mid, mid]*angstrom/kjmol).tolist(),
        }
        json.dump(jobj, jfile)

    plt.savefig(fn_png)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform a static potential energy scan.")
    parser.add_argument("input_fn", type=str,
                        help=".chk filename of the input structure")
    parser.add_argument("-fn_png", type=str, default="static_scan.png",
                        help=".png filename of the output figure")
    parser.add_argument("-node_idx", type=int, default=0,
                        help="index of the node to be moved around in the scanning range")
    parser.add_argument("-num", type=int, default=8,
                        help="number of scan locations in one dimension")

    args = parser.parse_args()
    main(
        args.input_fn,
        args.fn_png, 
        args.node_idx,
        args.num
    )
