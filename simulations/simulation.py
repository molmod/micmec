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

from trajectory import HDF5Writer, XYZWriter

from molmod.units import *


input_fn = "5x5x5_TEST.chk"

sys = System.from_file(input_fn)

# Give one node a slight deviation.
#pos = sys.pos.copy()
#pos[0, :] = np.array([2.0, 2.0, 2.0])*angstrom
#sys.pos = pos

mmf = MicroMechanicalField(sys)

timestep = 100.0*femtosecond
steps = 100

h5_fn = "testing.h5"

with h5py.File(h5_fn, mode = 'w') as h5_f:
    h5 = HDF5Writer(h5_f, step=1)
    vsl = VerletScreenLog(step=100)
    verlet = VerletIntegrator(mmf, timestep=timestep, hooks=[h5, vsl])

    verlet.run(steps)


