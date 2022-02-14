#!/usr/bin/env python
# File name: mmf.py
# Description: The calculation of forces acting on the micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

""" The calculation of forces acting on the nodes of the micromechanical model. """

import numpy as np

from log import log, timer

__all__ = ["MicroMechanicalField"]


class MicroMechanicalField(object):
    
    def __init__(self, system):

        self.system = system
        self.energy = 0.0
        self.gpos = np.zeros((self.system.nnodes, 3))
        self.clear()

        # Each node in a grid has eight neighboring cells.
        # The neighbors are indexed from 0 to 7 in a fixed order.
        neighbor_cells = [
            ( 0, 0, 0),
            (-1, 0, 0),
            ( 0,-1, 0),
            ( 0, 0,-1),
            (-1,-1, 0),
            (-1, 0,-1),
            ( 0,-1,-1),
            (-1,-1,-1)
        ]

        # Initialize the derivatives of the neighboring cell matrices to x, y and z.
        self.cell_xderivs = []
        self.cell_yderivs = []
        self.cell_zderivs = []

        # Compute the derivatives of the neighboring cell matrices to x, y and z.
        for neighbor_index, neighbor_cell in enumerate(neighbor_cells):
            
            cell_xderiv = np.zeros((3, 3))
            cell_yderiv = np.zeros((3, 3))
            cell_zderiv = np.zeros((3, 3))
            
            cell_xderiv[:, 0] = 0.25*np.array([1.0 if n == -1 else -1.0 for n in neighbor_cell])
            cell_yderiv[:, 1] = 0.25*np.array([1.0 if n == -1 else -1.0 for n in neighbor_cell])
            cell_zderiv[:, 2] = 0.25*np.array([1.0 if n == -1 else -1.0 for n in neighbor_cell])
            
            self.cell_xderivs.append(cell_xderiv)
            self.cell_yderivs.append(cell_yderiv)
            self.cell_zderivs.append(cell_zderiv)
    

    def clear(self):
        self.energy = np.nan
        self.gpos[:] = np.nan
        

    def update_pos(self, pos):
        self.clear()
        self.system.pos[:] = pos

    
    def compute(self, verlet_gpos):
        
        self.gpos[:] = 0.0
        self.energy = self._compute()
        verlet_gpos += self.gpos
        
        return self.energy
    
    
    def _compute(self):
        
        # Call all methods required to obtain the total potential energy.
        self._compute_cells()
        self._compute_strains()
        self._compute_gpos()
        
        # Initialize the energy.
        energy = 0.0
        for cell_index in range(self.system.ncells):
            
            # Get the cell matrix.
            cell = self.cell_matrices[cell_index]
            
            cell_det = np.linalg.det(cell)

            # Get the elasticity and strain tensors.
            # These are lists, in case there are multiple stable phases.
            elast_lst = self.system.elasticity_tensors[cell_index]
            strain_lst = self.strain_tensors[cell_index]
            # Iterate over all stable phases.
            for elast, strain in zip(elast_lst, strain_lst):
                # TO DO: generalize to multiple stable phases.
        
                # Compute the potential elastic energy of the cell and add it to the total.
                energy += 0.5*cell_det*np.tensordot(strain, np.tensordot(elast, strain, axes=2), axes=2)
        
        return energy

    
    def _compute_cells(self):
        """
        Compute the cell matrices from their neighboring nodes.
        All stable phases of a type have the same instantaneous cell matrix.

        Example:
            cell_matrices[cell_index]
            --> The cell matrix of the cell at location (kappa, lambda_, mu)
        
        """
        # Initialize empty lists to store the cell matrices and their inverses in.
        self.cell_matrices = []
        self.inv_cell_matrices = [] 

        # Compute the deviations of the nodes from their initial, rectangular positions in the grid.
        # This is a hack to deal with periodic boundary conditions.
        devs = self.system.pos - self.system.pos_ref
        
        for cell_index in range(self.system.ncells):
            
            # Store the deviations of the neighboring nodes in an array.
            node_devs = np.zeros((8, 3))
            for neighbor_index, node_index in self.system.surrounding_nodes[cell_index]:
                node_devs[neighbor_index] = devs[node_index, :]
            
            xvec = self.system.cell_ref[cell_index, 0].copy()
            yvec = self.system.cell_ref[cell_index, 1].copy()
            zvec = self.system.cell_ref[cell_index, 2].copy()
            
            xvec += 0.25*(-node_devs[0] + node_devs[1] - node_devs[2] + node_devs[4] \
                        - node_devs[3] + node_devs[5] - node_devs[6] + node_devs[7])

            yvec += 0.25*(-node_devs[0] + node_devs[2] - node_devs[1] + node_devs[4] \
                        - node_devs[3] + node_devs[6] - node_devs[5] + node_devs[7])

            zvec += 0.25*(-node_devs[0] + node_devs[3] - node_devs[2] + node_devs[6] \
                        - node_devs[1] + node_devs[5] - node_devs[4] + node_devs[7])
            
            # Construct the cell matrix.
            cell = np.array([xvec.copy(), yvec.copy(), zvec.copy()])
            
            # Store the cell matrices and the inverse cell matrices.
            self.cell_matrices.append(cell.copy())
            self.inv_cell_matrices.append(np.linalg.inv(cell))
    
    
    def _compute_strains(self):
        """
        Compute the strain tensors from the cell matrices and the equilibrium inverse cell matrices.
        Separate stable phases have different strain tensors!

        Example:
            strain_tensors[cell_index][0]
            --> This is the strain tensor of the FIRST stable phase of the cell at location (kappa, lambda_, mu).
        """
        # Initialize an empty list to store the strain tensors in.
        # Separate phases can have different strain tensors!
        self.strain_tensors = []
        
        for cell_index in range(self.system.ncells):
            
            # Initialize a strain tensor.
            strain = np.zeros((3, 3))
            
            cell = self.cell_matrices[cell_index]
            cell0_inv_lst = self.system.equilibrium_inv_cell_matrices[cell_index]
            
            strain_lst = []
            for cell0_inv in cell0_inv_lst:
                strain = 0.5*((cell @ cell0_inv).T @ (cell @ cell0_inv) - np.identity(n=3))
                strain_lst.append(strain)
            
            self.strain_tensors.append(strain_lst)

        
    def _compute_gpos(self):
        """
        Compute the gradient of the potential energy for each node in the network.
        The output has the same shape as the array of positions, (N, 3).
        """
        # Iterate over each node
        for node_index in range(self.system.nnodes):
            
            fx_node = 0.0
            fy_node = 0.0
            fz_node = 0.0

            # Iterate over each neighboring cell of node (k, l, m).
            for neighbor_index, cell_index in self.system.surrounding_cells[node_index]:

                # Get the derivatives to x, y and z of the cell matrix.
                cell_xderiv = self.cell_xderivs[neighbor_index]
                cell_yderiv = self.cell_yderivs[neighbor_index]
                cell_zderiv = self.cell_zderivs[neighbor_index]

                cell = self.cell_matrices[cell_index]
                cell_inv = self.inv_cell_matrices[cell_index]

                cell_xtrace = np.trace(cell_inv @ cell_xderiv)
                cell_ytrace = np.trace(cell_inv @ cell_yderiv)
                cell_ztrace = np.trace(cell_inv @ cell_zderiv)
                cell_det = np.linalg.det(cell)

                # These have been calculated before.
                # They are lists, in case there are multiple stable phases.
                cell0_lst = self.system.equilibrium_cell_matrices[cell_index]
                cell0_inv_lst = self.system.equilibrium_inv_cell_matrices[cell_index]
                elast_lst = self.system.elasticity_tensors[cell_index]
                
                strain_lst = self.strain_tensors[cell_index]
                
                # Iterate over all stable phases.
                for cell0, cell0_inv, elast, strain in zip(cell0_lst, cell0_inv_lst, elast_lst, strain_lst):
                    # TO DO: generalize to multiple stable phases.                  

                    # Compute the derivatives to x, y and z of the strain tensor.
                    _xmat = cell0_inv.T @ cell.T @ cell_xderiv @ cell0_inv
                    _ymat = cell0_inv.T @ cell.T @ cell_yderiv @ cell0_inv
                    _zmat = cell0_inv.T @ cell.T @ cell_zderiv @ cell0_inv
                    
                    strain_xderiv = 0.5*(_xmat.T + _xmat) # EQ.3
                    strain_yderiv = 0.5*(_ymat.T + _ymat) # EQ.3
                    strain_zderiv = 0.5*(_zmat.T + _zmat) # EQ.3
                    
                    # Compute the contribution of cell (kappa, lamnda, mu) to the x component
                    # of the force acting on node (k, l, m).
                    fx = cell_xtrace*np.tensordot(strain.T, np.tensordot(elast, strain, axes=2), axes=2) # EQ.2 first term
                    fx += np.tensordot(strain.T, np.tensordot(elast, strain_xderiv, axes=2), axes=2) # EQ.2 second term
                    fx += np.tensordot(strain_xderiv.T, np.tensordot(elast, strain, axes=2), axes=2) # EQ.2 third term
                    
                    # Compute the contribution of cell (kappa, lamnda, mu) to the y component
                    # of the force acting on node (k, l, m).
                    fy = cell_ytrace*np.tensordot(strain.T, np.tensordot(elast, strain, axes=2), axes=2) # EQ.2 first term
                    fy += np.tensordot(strain.T, np.tensordot(elast, strain_yderiv, axes=2), axes=2) # EQ.2 second term
                    fy += np.tensordot(strain_yderiv.T, np.tensordot(elast, strain, axes=2), axes=2) # EQ.2 third term
                    
                    # Compute the contribution of cell (kappa, lamnda, mu) to the y component
                    # of the force acting on node (k, l, m).
                    fz = cell_ztrace*np.tensordot(strain.T, np.tensordot(elast, strain, axes=2), axes=2) # EQ.2 first term
                    fz += np.tensordot(strain.T, np.tensordot(elast, strain_zderiv, axes=2), axes=2) # EQ.2 second term
                    fz += np.tensordot(strain_zderiv.T, np.tensordot(elast, strain, axes=2), axes=2) # EQ.2 third term
                    
                    # Scale the contribution of cell (kappa, lambda, mu) according to the volume of the cell.
                    fx *= -0.5*cell_det # EQ.2 prefactor
                    fy *= -0.5*cell_det # EQ.2 prefactor
                    fz *= -0.5*cell_det # EQ.2 prefactor
                    

                # Add the contribution of cell (kappa, lambda, mu) to the total force acting on node (k, l, m).               
                fx_node += fx
                fy_node += fy
                fz_node += fz
            
            # Combine the force components into a vector.
            self.gpos[node_index, :] = -np.array([fx_node, fy_node, fz_node])
    



