#!/usr/bin/env python
# File name: test.py
# Description: Testing the micromechanical model.
# Author: Joachim Vandewalle
# Date: 20-11-2021

import numpy as np

import matplotlib.pyplot as plt

from molmod.units import *
from molmod.constants import *

from system import *
from mmf import *
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

fig = plt.figure()
ax = plt.axes()
fns = ["2x2x2_TEST.chk"]
for ind, fn in enumerate(fns):
    # The input file for this toy system has been stored previously in "2x2x2_TEST.chk".
    # Initialize the system and the micromechanical field.
    sys = System.from_file(fn)
    mmf = MicroMechanicalField(sys)

    pos = sys.pos.copy()
    pos_ref = sys.pos_ref.copy()

    # It is preferred that the scan has an even number of scan locations (num) in each dimension.
    # After taking the numerical derivative, then it will be possible to obtain values exactly in the center
    # of the scanning range if num is even.
    num = 8
    mid = (num - 2)//2

    # Define a maximum deviation of the node in each dimension, in atomic units.
    # This should not exceed the dimensions of the cell (10*angstrom).
    max_dev = 0.25*unit_cell_length

    # Scanning ranges.
    x_range = max_dev*np.linspace(-1, 1, num)
    y_range = max_dev*np.linspace(-1, 1, num)
    z_range = max_dev*np.linspace(-1, 1, num)

    # The scan will evaluate the ENERGY_POT and the forces at each scan location (x, y, z).
    X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing="ij")

    ENERGY_POT = np.zeros((num, num, num))
    ZERO = np.zeros((num, num, num))
    FX = np.zeros((num, num, num))
    FY = np.zeros((num, num, num))
    FZ = np.zeros((num, num, num))

    NODE_INDEX = 2

    for SCAN_INDEX, _ in np.ndenumerate(ZERO):
        gpos = np.zeros(pos.shape)
        # Manually move one node to the current scanning location.
        pos[NODE_INDEX] = pos_ref[NODE_INDEX] + np.array([X[SCAN_INDEX], Y[SCAN_INDEX], Z[SCAN_INDEX]])
        mmf.update_pos(pos)
        ENERGY_POT[SCAN_INDEX] = mmf.compute(gpos)
        FX[SCAN_INDEX] = -gpos[NODE_INDEX, 0]
        FY[SCAN_INDEX] = -gpos[NODE_INDEX, 1]
        FZ[SCAN_INDEX] = -gpos[NODE_INDEX, 2]

    FX_numerical = -np.diff(ENERGY_POT, axis=0)/np.diff(X, axis=0)
    FY_numerical = -np.diff(ENERGY_POT, axis=1)/np.diff(Y, axis=1)
    FZ_numerical = -np.diff(ENERGY_POT, axis=2)/np.diff(Z, axis=2)

    FX_analytical = 0.5*(FX[:-1, :, :] + FX[1:, :, :])
    FY_analytical = 0.5*(FY[:, :-1, :] + FY[:, 1:, :])
    FZ_analytical = 0.5*(FZ[:, :, :-1] + FZ[:, :, 1:])

    X_ = 0.5*(X[:-1, :, :] + X[1:, :, :])
    Y_ = 0.5*(Y[:, :-1, :] + Y[:, 1:, :])
    Z_ = 0.5*(Z[:, :, :-1] + Y[:, :, 1:])

""" 
ax = plt.axes(projection='3d')
ax.plot_surface(X[:, :, mid]/angstrom, Y[:, :, mid]/angstrom, ENERGY_POT[:, :, mid]/kjmol, cmap="viridis")
#    if ind == 0:
#        ax.contour(X[:, :, mid]/angstrom, Y[:, :, mid]/angstrom, ENERGY_POT[:, :, mid]/kjmol, linestyles="--")
#        ax.plot([0, 0], [0, 0], color="black", linestyle="--", label="asymmetrical system")
#    else:
#        ax.contour(X[:, :, mid]/angstrom, Y[:, :, mid]/angstrom, ENERGY_POT[:, :, mid]/kjmol)
#        ax.plot([0, 0], [0, 0], color="black", linestyle="-", label="symmetrical system")


ax.set_xlabel("$x - x_0$ [Å]")
ax.set_ylabel("$y - y_0$ [Å]")
ax.set_zlabel("POTENTIAL ENERGY [kJ/mol]")
#ax.set_zlim(0.0, 18.0)
#plt.axis("square")
#plt.legend()
plt.show()
"""
plt.plot(X_[:, mid, mid]/angstrom, FX_numerical[:, mid, mid]*angstrom/kjmol, label="numerical derivative", color="orange")
plt.plot(X_[:, mid, mid]/angstrom, FX_analytical[:, mid, mid]*angstrom/kjmol, linewidth=0, marker="x", label="analytical expression", color="blue")
plt.xlabel("$x - x_0$ [Å]")
plt.ylabel("$f_x$ [kJ/mol/Å]")
plt.legend()
plt.grid()
plt.show()

rico, intercept = np.polyfit(X_[:, mid, mid], FX_numerical[:, mid, mid], deg=1)
force_con = -rico
timestep = np.pi*np.sqrt(mass/force_con)
elasticity_max = np.amax(elasticity_matrix)
timestep_est = np.pi*np.sqrt(mass/(elasticity_max*unit_cell_length))
print(f"TIMESTEP FROM FORCE CONSTANT: {timestep/femtosecond} fs")
print(f"TIMESTEP FROM ESTIMATION: {timestep_est/femtosecond} fs")



