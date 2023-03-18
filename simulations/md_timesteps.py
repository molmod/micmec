#!/usr/bin/env python
# File name: md_timesteps.py
# Description: Perform multiple MD simulations with different timesteps.
# Author: Joachim Vandewalle
# Date: 20-11-2021

import numpy as np

import h5py
import argparse

from micmec.log import log
from micmec.system import System
from micmec.pes.mmff import MicMecForceField, ForcePartMechanical
from micmec.sampling.verlet import VerletIntegrator, VerletScreenLog

from micmec.sampling.trajectory import HDF5Writer
from molmod.units import *


def main(input_fn, total_time, lower_bound, upper_bound):
    timesteps = np.linspace(lower_bound, upper_bound, 5)*femtosecond

    for idx, timestep in enumerate(timesteps):

        steps = int(total_time*picosecond/timestep)
        sys = System.from_file(input_fn)
        fpm = ForcePartMechanical(sys)
        mmf = MicMecForceField(sys, [fpm])

        h5_fn = f"timestep{idx}.h5"

        try:
            with h5py.File(h5_fn, mode = 'w') as h5_f:
                h5 = HDF5Writer(h5_f, step=1)
                vsl = VerletScreenLog(step=1)
                if idx == 0:
                    verlet = VerletIntegrator(mmf, timestep=timestep, hooks=[h5, vsl])
                    random_vel = verlet.vel.copy()
                else:
                    verlet = VerletIntegrator(mmf, timestep=timestep, vel0=random_vel, hooks=[h5])

                verlet.run(steps)
        except FloatingPointError:
            log("Simulation has diverged!")
        else:
            log("Simulation has converged.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform a micromechanical MD simulation with varying timesteps."
    )
    parser.add_argument("input_fn", type=str, help=".chk filename of the input structure")
    parser.add_argument("-total_time", type=float, default=10.0, help="total time [ps]")
    parser.add_argument("-lower_bound", type=float, default=50.0, help="lower bound for the timestep [fs]")
    parser.add_argument("-upper_bound", type=float, default=250.0, help="upper bound for the timestep [fs]")
    args = parser.parse_args()
    main(args.input_fn, args.total_time, args.lower_bound, args.upper_bound)

