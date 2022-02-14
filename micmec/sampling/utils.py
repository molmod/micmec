#!/usr/bin/env python
# File name: utils.py
# Description: Useful functions for use in a micromechanical simulation.
# Author: Joachim Vandewalle
# Date: 26-10-2021

from molmod import boltzmann

import numpy as np

def get_random_vel(temp0, scalevel0, masses, select=None):
    if select is not None:
        masses = masses[select]
    shape_result = (len(masses), 3)
    result = np.random.normal(0, 1, shape_result)*np.sqrt(boltzmann*temp0/masses).reshape(-1,1)
    if scalevel0 and temp0 > 0:
        temp = np.mean((result**2*masses.reshape(-1,1)))/boltzmann
        scale = np.sqrt(temp0/temp)
        result *= scale
    return result

def remove_com_moment(vel, masses):
    # Compute the center of mass velocity.
    com_vel = np.dot(masses, vel)/np.sum(masses)
    # Subtract this com velocity vector from each node velocity.
    vel[:] -= com_vel
