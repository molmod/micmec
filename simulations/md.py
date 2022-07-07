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
import argparse

from micmec.system import System
from micmec.pes.mmff import MicMecForceField, ForcePartMechanical
from micmec.sampling.verlet import VerletIntegrator, VerletScreenLog
from micmec.sampling.trajectory import HDF5Writer, XYZWriter
from micmec.sampling.nvt import NHCThermostat, LangevinThermostat
from micmec.sampling.npt import MTKBarostat, TBCombination, LangevinBarostat

from molmod.units import kelvin, pascal, femtosecond


def main(input_fn, output_fn, timestep, steps, temp, press, file_step, log_step):
    
    # Define the system and the micromechanical force field.
    sys = System.from_file(input_fn)
    fpm = ForcePartMechanical(sys)
    mmf = MicMecForceField(sys, [fpm])

    timestep *= femtosecond
    
    with h5py.File(output_fn, mode = "w") as f:
        h5 = HDF5Writer(f, step=file_step)
        vsl = VerletScreenLog(step=log_step)
        if (temp is None) and (press is None):
            # (N, V, E) ensemble.
            verlet = VerletIntegrator(mmf, timestep=timestep, hooks=[h5, vsl])
        elif (temp is not None) and (press is None):
            # (N, V, T) ensemble.
            temp *= kelvin
            # Define the thermostat.
            lt = LangevinThermostat(temp=temp, timecon=100*timestep)
            verlet = VerletIntegrator(mmf, timestep=timestep, hooks=[h5, vsl, lt])
        elif (temp is not None) and (press is not None):
            # (N, P, T) ensemble.
            temp *= kelvin
            press *= (1e6)*pascal   
            # Define the thermostat-barostat combination.
            lt = LangevinThermostat(temp=temp, timecon=100*timestep)
            lb = LangevinBarostat(mmf, temp=temp, press=press, timecon=1000*timestep)
            tbc = TBCombination(lt, lb)
            verlet = VerletIntegrator(mmf, timestep=timestep, hooks=[h5, vsl, tbc])
        
        verlet.run(steps)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Perform a micromechanical MD simulation.")
    parser.add_argument("input_fn", type=str,
                        help=".chk filename of the input structure")
    parser.add_argument("output_fn", type=str,
                        help=".h5 filename of the output trajectory")
    parser.add_argument("-timestep", type=int, default=10,
                        help="timestep [fs]")
    parser.add_argument("-steps", type=int, default=100,
                        help="integer number of steps to simulate")
    parser.add_argument("-temp", type=float, default=None,
                        help="thermostat temperature [K]")
    parser.add_argument("-press", type=float, default=None,
                        help="barostat pressure [MPa]")
    parser.add_argument("-file_step", type=int, default=1,
                        help="step of the trajectory recorded in the .h5 file")
    parser.add_argument("-log_step", type=int, default=10,
                        help="step of the screen logger")

    args = parser.parse_args()
    main(args.input_fn,
        args.output_fn, 
        args.timestep, 
        args.steps, 
        args.temp, 
        args.press, 
        args.file_step,
        args.log_step)


