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

"""Perform an optimization of the micromechanical nodes."""

import numpy as np

import h5py
import argparse

from micmec.system import System
from micmec.pes.mmff import MicMecForceField, ForcePartMechanical
from micmec.sampling.opt import CGOptimizer, OptScreenLog, QNOptimizer
from micmec.sampling.dof import CartesianDOF

from micmec.sampling.trajectory import HDF5Writer


def main(input_fn, output_fn, file_step, log_step):
    # Define the system and the micromechanical force field.
    sys = System.from_file(input_fn)
    fpm = ForcePartMechanical(sys)
    mmf = MicMecForceField(sys, [fpm])
    # Perform only an optimisation of the nodal positions, not of the domain parameters.
    cdof = CartesianDOF(mmf)
    
    with h5py.File(output_fn, mode = "w") as f:
        h5 = HDF5Writer(f, step=file_step)
        osl = OptScreenLog(step=log_step)
        qnopt = QNOptimizer(cdof, hooks=[h5, osl])
        qnopt.run()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform a micromechanical optimisation.")
    parser.add_argument("input_fn", type=str,
                        help=".chk filename of the input structure")
    parser.add_argument("output_fn", type=str,
                        help=".h5 filename of the output trajectory")
    parser.add_argument("-file_step", type=int, default=1,
                        help="step of the trajectory recorded in the .h5 file")
    parser.add_argument("-log_step", type=int, default=1,
                        help="step of the screen logger")

    args = parser.parse_args()
    main(
        args.input_fn,
        args.output_fn, 
        args.file_step,
        args.log_step
    )




