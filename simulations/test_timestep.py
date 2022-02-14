#!/usr/bin/env python
# File name: test.py
# Description: Testing the micromechanical model.
# Author: Joachim Vandewalle
# Date: 20-11-2021

import numpy as np
import h5py

import matplotlib.pyplot as plt

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

timesteps = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

colors = ["#e41a1c", "#377eb8", "#4daf4a", 
               "#984ea3", "#ff7f00", "#ffff33", "#a65628",
               "#f781bf", "#999999", "#66c2a5", "#fc8d62",
               "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
               "#e5c494", "#b3b3b3", "#8dd3c7", "#ffffb3",
               "#bebada", "#fb8072", "#80b1d3", "#fdb462",
               "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
               "#ccebc5", "#ffed6f"]

for idx, timestep in enumerate(timesteps[:]):
    
    h5_fn = f"timestep{idx}_fcu_reo_trajectory.h5"

    with h5py.File(h5_fn, mode = 'r') as h5_f:
        
        dset_traj_energy = h5_f['trajectory/etot']
        dset_traj_energy_pot = h5_f['trajectory/epot']
        dset_traj_energy_kin = h5_f['trajectory/ekin']
        dset_traj_cons_err = h5_f['trajectory/cons_err']
                
        # Copy datasets to arrays.
        traj_energy = np.array(dset_traj_energy)
        traj_energy_pot = np.array(dset_traj_energy_pot)
        traj_energy_kin = np.array(dset_traj_energy_kin)
        traj_cons_err = np.array(dset_traj_cons_err)

        time = timestep*np.arange(len(traj_energy))*femtosecond
        
        plt.plot(time/nanosecond, traj_energy/kjmol, label=f"{timestep} fs", color=colors[idx], linestyle="-")
        #plt.plot(time/picosecond, traj_energy_kin/kjmol, label=f"KINETIC ENERGY", color=colors[idx], linestyle=":")
        #plt.plot(time/picosecond, traj_energy_pot/kjmol, label=f"POTENTIAL ENERGY", color=colors[idx], linestyle="--")
        
        #plt.plot(time/nanosecond, traj_energy[0]*np.ones(len(traj_energy))/electronvolt, color=colors[idx])

#timestep = 100.0
#steps = int(1.0*nanosecond/(timestep*femtosecond))
#h5_fn = "fcu_reo_trajectory.h5"
#idx = 0
#
#with h5py.File(h5_fn, mode = 'r') as h5_f:
#    
#    dset_traj_energy = h5_f['trajectory/etot']
#    dset_traj_energy_pot = h5_f['trajectory/epot']
#    dset_traj_energy_kin = h5_f['trajectory/ekin']
#    dset_traj_cons_err = h5_f['trajectory/cons_err']
#            
#    # Copy datasets to arrays.
#    traj_energy = np.array(dset_traj_energy)
#    traj_energy_pot = np.array(dset_traj_energy_pot)
#    traj_energy_kin = np.array(dset_traj_energy_kin)
#    traj_cons_err = np.array(dset_traj_cons_err)
#
#    time = 10*timestep*np.arange(len(traj_energy))*femtosecond
#    
#    plt.plot(time/picosecond, traj_energy/kjmol, label=f"TOTAL ENERGY", color=colors[idx], linestyle="-")
#    plt.plot(time/picosecond, traj_energy_kin/kjmol, label=f"KINETIC ENERGY", color=colors[idx], linestyle=":")
#    plt.plot(time/picosecond, traj_energy_pot/kjmol, label=f"POTENTIAL ENERGY", color=colors[idx], linestyle="--")
#    
#    #plt.plot(time/nanosecond, traj_energy[0]*np.ones(len(traj_energy))/electronvolt, color=colors[idx])
#
plt.xlabel("TIME [ps]")
plt.ylabel("ENERGY [kJ/mol]")
plt.xlim(0.0, 0.01)
plt.ylim(0.0, 500.0)
plt.legend()
plt.grid()
plt.show()



