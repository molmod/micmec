#!/usr/bin/env python
# File name: test.py
# Description: Testing the micromechanical model.
# Author: Joachim Vandewalle
# Date: 20-11-2021

import numpy as np
import h5py

from system import System
from mmf import MicroMechanicalField
from verlet import VerletIntegrator, VerletScreenLog

from trajectory import HDF5Writer

from molmod.units import *
from tensor import *

gigapascal = (10**9)*pascal

# Define a toy system with very easy parameters in atomic units.
mass = 10000000
unit_cell_length = 10.0*angstrom
cell = np.array([[unit_cell_length, 0.0, 0.0],
                 [0.0, unit_cell_length, 0.0],
                 [0.0, 0.0, unit_cell_length]])
elasticity_matrix = np.array([[50.0, 30.0, 30.0,  0.0,  0.0,  0.0],
                              [30.0, 50.0, 30.0,  0.0,  0.0,  0.0],
                              [30.0, 30.0, 50.0,  0.0,  0.0,  0.0],
                              [ 0.0,  0.0,  0.0, 20.0,  0.0,  0.0],
                              [ 0.0,  0.0,  0.0,  0.0, 20.0,  0.0],
                              [ 0.0,  0.0,  0.0,  0.0,  0.0, 20.0]])*gigapascal
elasticity_tensor = voigt_inv(elasticity_matrix, mode="elasticity")

# The input file for this toy system has been stored previously in "2x2x2_TEST.chk".
# Initialize the system and the micromechanical field.

timesteps = np.array([100.0, 200.0, 300.0, 400.0, 500.0])*femtosecond

for idx, timestep in enumerate(timesteps):

    steps = int(0.01*nanosecond/timestep)
    sys = System.from_file("3x3x3_demo.chk")
    mmf = MicroMechanicalField(sys)

    h5_fn = f"timestep{idx}_fcu_reo_trajectory.h5"

    with h5py.File(h5_fn, mode = 'w') as h5_f:
        h5 = HDF5Writer(h5_f, step=1)
        if idx == 0:
            verlet = VerletIntegrator(mmf, timestep=timestep, hooks=[h5])
            random_vel = verlet.vel.copy()
        else:
            verlet = VerletIntegrator(mmf, timestep=timestep, vel0=random_vel, hooks=[h5])

        verlet.run(steps)


