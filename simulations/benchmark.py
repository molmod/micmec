#!/usr/bin/env python
# File name: simulation.py
# Description: Run a molecular dynamics simulation using the micromechanical model.
# Author: Joachim Vandewalle
# Date: 26-10-2021

import numpy as np
import h5py

from system import System
from mmf import MicroMechanicalField
from verlet import VerletIntegrator, VerletScreenLog

from trajectory import HDF5Writer

from molmod.units import *


input_fns = ["2x2x2_TEST.chk", "3x3x3_TEST.chk", "4x4x4_TEST.chk", "5x5x5_TEST.chk"]

for input_fn in input_fns:

    sys = System.from_file(input_fn)
    mmf = MicroMechanicalField(sys)

    timestep = 100.0*femtosecond
    steps = 1000

    h5_fn = input_fn[:5] + "_TEST_trajectory.h5"

    with h5py.File(h5_fn, mode = 'w') as h5_f:
        h5 = HDF5Writer(h5_f, step=10)
        vsl = VerletScreenLog(step=100)
        verlet = VerletIntegrator(mmf, timestep=timestep, hooks=[h5, vsl])

        verlet.run(steps)


