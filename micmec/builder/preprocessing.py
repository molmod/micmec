#!/usr/bin/env python
# File name: processing.py
# Description: Extracting elastic properties of a nanocell.
# Author: Joachim Vandewalle
# Date: 26-10-2021

""" Processing the output of one or multiple NPT MD simulation to extract the elastic properties of the nanocell. """

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import h5py
import yaff
import sys

import micmec.analysis.tensor as tensor

from molmod.units import *
from molmod.constants import *

__all__ = ["Nanocell", "SimulationProperties", "ElasticProperties"]

gigapascal = (10**9)*pascal


class Nanocell(object):

    def __init__(self, material, mass, topology=None, effective_temp=None):
        """This class holds the characteristics of a nanocell in the micromechanical model."""
        self.material = material
        if topology is None:
            self.topology = "UNKNOWN"
        else:
            self.topology = topology
        self.mass = mass
        self.elasticity = []
        self.cell = []
        self.free_energy = []
        if effective_temp is None:
            self.effective_temp = 300*kelvin
        else:
            self.effective_temp = effective_temp
            
            
    def add_state(self, cell, elasticity, free_energy=None):
        
        self.cell.append(cell)
        self.elasticity.append(elasticity)
        if free_energy is None:
            self.free_energy.append(0.0)
        else:
            self.free_energy.append(free_energy)
        
    
    def to_pkl(self, output_fn):
        """ 
        Create a pickle file which contains the elastic properties of one type of nanocell.
        The properties of the type, as they first appear in a simulation, are always the first.
        """
        output = {}

        # The following variables are general properties of the nanocell.
        output["material"] = self.material
        output["topology"] = self.topology
        output["mass"] = self.mass
        output["cell"] = self.cell
        output["elasticity"] = self.elasticity
        output["free_energy"] = self.free_energy
        output["effective_temp"] = self.effective_temp

        with open(output_fn, "wb") as pklf:
            pkl.dump(output, pklf)



class ElasticProperties(object):

    def __init__(self, cellt, temp0):
        """
        Calculate the elastic properties from the time series of a cell matrix and its equilibrium temperature
        in the (N, P, T) ensemble.
        """
        self.temp0 = temp0
        self.cellt = cellt
        
        # Calculating the equilibrium cell matrix.
        self.cell0 = np.mean(self.cellt, axis=0) 
        self.cell0_inv = np.linalg.inv(self.cell0)

        # Calculating the equilibrium volume and strain.
        self.volumet = np.array([self.compute_volume(cell) for cell in self.cellt])
        self.volume0 = np.linalg.det(self.cell0) # Instead of taking the mean of the volume time series.
        self.straint = np.array([self.compute_strain(cell, self.cell0_inv) for cell in self.cellt])
        self.strain0 = np.mean(self.straint, axis=0)

        # Calculating the equilibrium compliance tensor.
        self.compliancet = np.array([self.compute_compliance(strain, self.strain0, self.volume0, self.temp0) for strain in self.straint])
        self.compliance0 = np.mean(self.compliancet, axis=0)       

        # Constructing the compliance matrix (Voigt notation).
        self.compliance0_matrix = tensor.voigt(self.compliance0, mode="compliance")

        # Obtaining the elasticity matrix by inversion.
        self.elasticity0_matrix = np.linalg.inv(self.compliance0_matrix)

        # Constructing the elasticity tensor from the elasticity matrix.
        self.elasticity0 = tensor.voigt_inv(self.elasticity0_matrix, mode="elasticity")

        # Constructing the identity tensor. (DEBUGGING)
        self.ident0 = np.zeros((3,3,3,3))
        for index, _ in np.ndenumerate(self.ident0):
            self.ident0[index] = 0.5*((index[0] == index[2])*(index[1] == index[3]) 
                                    + (index[1] == index[2])*(index[0] == index[3]))

        self.ident1 = np.tensordot(self.elasticity0, self.compliance0, axes=2)
    

    def compute_volume(self, cell):
        return np.linalg.det(cell)

    def compute_strain(self, cell, cell0_inv):
        return 0.5*((cell0_inv @ cell) @ (cell0_inv @ cell).T - np.identity(n=3))    

    def compute_compliance(self, strain, strain0, volume0, temp0):
        return (volume0/(boltzmann*kelvin*temp0))*np.tensordot(strain-strain0, strain-strain0, axes=0)
    

    def get_compliance_tensor(self):
        return self.compliance0

    def get_compliance_matrix(self):
        return self.compliance0_matrix
    
    def get_elasticity_tensor(self):
        return self.elasticity0

    def get_elasticity_matrix(self):
        return self.elasticity0_matrix

    def get_cell_matrix(self):
        return self.cell0



class SimulationProperties(object):
    
    def __init__(self, input_fn, time_step=0.5*femtosecond):
        """
        Collect the properties of an MD simulation from its HDF5 file.
        """
        if input_fn.endswith(".h5"):
            with h5py.File(input_fn, mode = 'r') as h5f:
        
                dset_sys_masses = h5f['system/masses']
                
                # The 'trajectory/cell' dataset is a time series of the cell matrix.
                dset_traj_cell = h5f['trajectory/cell']
                dset_traj_temp = h5f['trajectory/temp']
                dset_traj_press = h5f['trajectory/press']
                
                # Copy datasets to arrays.
                traj_cell = np.array(dset_traj_cell)
                traj_temp = np.array(dset_traj_temp)
                traj_press = np.array(dset_traj_press)

                sys_masses = np.array(dset_sys_masses)
        else:
            raise IOError("Cannot read from file \'%s\'." % input_fn)

        self.time_step = time_step
        
        L = len(traj_cell)
        
        self.cellt = traj_cell[L//2:]
        self.tempt = traj_temp[L//2:]
        self.presst = traj_press[L//2:]

        L_ = len(self.cellt)
        
        self.mass = np.sum(sys_masses)
        self.time = np.arange(L_)*time_step
        
        self.temperature = np.mean(self.tempt, axis=0)
        self.pressure = np.mean(self.presst, axis=0)

        self.elastic_properties = ElasticProperties(self.cellt, self.temperature)
        self.elasticity = self.elastic_properties.get_elasticity_tensor()
        self.cell = self.elastic_properties.get_cell_matrix()
        


def main():
    # Define a toy system with very easy parameters in atomic units.
    mass = 10000000
    unit_cell_length = 10.0*angstrom
    cell_0 = np.array([[unit_cell_length, 0.0, 0.0],
                        [0.0, unit_cell_length, 0.0],
                        [0.0, 0.0, unit_cell_length]])
    cell_1 = np.array([[0.5*unit_cell_length, 0.0, 0.0],
                        [0.0, 2.0*unit_cell_length, 0.0],
                        [0.0, 0.0, 2.0*unit_cell_length]])
    elasticity_matrix = np.array([[20.0, 10.0, 10.0,  0.0,  0.0,  0.0],
                                  [10.0, 20.0, 10.0,  0.0,  0.0,  0.0],
                                  [10.0, 10.0, 20.0,  0.0,  0.0,  0.0],
                                  [ 0.0,  0.0,  0.0, 10.0,  0.0,  0.0],
                                  [ 0.0,  0.0,  0.0,  0.0, 10.0,  0.0],
                                  [ 0.0,  0.0,  0.0,  0.0,  0.0, 10.0]])*gigapascal
    elasticity_tensor = tensor.voigt_inv(elasticity_matrix, mode="elasticity")
    
    test_cell_0 = Nanocell(material="new_test_0", mass=mass, topology="test")
    test_cell_0.add_state(cell=cell_0, elasticity=elasticity_tensor)
    test_cell_0.to_pkl("new_test_0.pickle")
    
    test_cell_1 = Nanocell(material="new_test_1", mass=mass, topology="test")
    test_cell_1.add_state(cell=cell_1, elasticity=elasticity_tensor)
    test_cell_1.to_pkl("new_test_1.pickle")

if __name__ == '__main__':
    main()


