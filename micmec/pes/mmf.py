#!/usr/bin/env python
# File name: mmf.py
# Description: MicroMechanicalField. The calculation of forces acting on the micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

"""The calculation of forces acting on the nodes of the micromechanical model."""

import numpy as np

from molmod import boltzmann

from micmec.log import log, timer

__all__ = [
    "MicMecForceField", 
    "ForcePart", 
    "ForcePartMechanical", 
    "ForcePartPressure"
]

class ForcePart(object):
    """
    Base class for anything that can compute energies (and optionally 
    gradient and virial) for a `System` object.
    """
    def __init__(self, name, system):
        """
        **ARGUMENTS**
        name
            A name for this part of the micromechanical force field. 
            This name must adhere to the following conventions: all lower case, 
            no white space, and short. It is used to construct part_* 
            attributes in the MicMecForceField class, where * is the name.
        system
            The system to which this part of the MMFF applies.
        """
        self.name = name
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.clear()

    def clear(self):
        """
        Fill in nan values in the cached results to indicate that they have become invalid.
        """
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):
        """
        Let the `ForcePart` object know that the domain vectors have changed.

        **ARGUMENTS**
        rvecs
            The new domain vectors.
        """
        self.clear()

    def update_pos(self, pos):
        """
        Let the `ForcePart` object know that the nodal positions have changed.

        **ARGUMENTS**
        pos
            The new nodal coordinates.
        """
        self.clear()

    def compute(self, gpos=None, vtens=None):
        """
        Compute the energy of this part of the MicMecForceField.
        
        The only variable inputs for the compute routine are the nodal
        positions and the domain vectors, which can be changed through the
        `update_rvecs` and `update_pos` methods. All other aspects of
        the micromechanical field are considered to be fixed between subsequent 
        compute calls. If changes other than positions or domain vectors are needed,
        one must construct a new MicMecForceField instance.
        
        **OPTIONAL ARGUMENTS**
        gpos
            The derivatives of the energy towards the Cartesian coordinates
            of the nodes. ("g" stands for gradient and "pos" for positions.)
            This must be a writeable numpy array with shape (N, 3) where N
            is the number of nodes.
        vtens
            The force contribution to the pressure tensor. This is also
            known as the virial tensor. It represents the derivative of the
            energy towards uniform deformations, including changes in the
            shape of the unit domain. (v stands for virial and "tens" stands
            for tensor.) This must be a writeable numpy array with shape (3,
            3). Note that the factor 1/V is not included.
        
        The energy is returned. The optional arguments are Fortran-style
        output arguments. When they are present, the corresponding results
        are computed and **added** to the current contents of the array.
        """
        if gpos is None:
            mmff_gpos = None
        else:
            mmff_gpos = self.gpos
            mmff_gpos[:] = 0.0
        if vtens is None:
            mmff_vtens = None
        else:
            mmff_vtens = self.vtens
            mmff_vtens[:] = 0.0
        
        self.energy = self._internal_compute(mmff_gpos, mmff_vtens)
        
        if np.isnan(self.energy):
            raise ValueError("The energy is not-a-number (nan).")
        if gpos is not None:
            if np.isnan(mmff_gpos).any():
                raise ValueError("Some gpos element(s) is/are not-a-number (nan).")
            gpos += mmff_gpos
        if vtens is not None:
            if np.isnan(mmff_vtens).any():
                raise ValueError("Some vtens element(s) is/are not-a-number (nan).")
            vtens += mmff_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        """Subclasses implement their compute code here."""
        raise NotImplementedError



class MicMecForceField(ForcePart):
    """A complete micromechanical force field model."""
    def __init__(self, system, parts):
        """
        **ARGUMENTS**
        system
            An instance of the `System` class.
        parts
            A list of instances of subclasses of `ForcePart`. These are
            the different types of contributions to the micromechanical force field.
        """
        ForcePart.__init__(self, "all", system)
        self.system = system
        self.parts = []
        for part in parts:
            self.add_part(part)
        if log.do_medium:
            with log.section("MMFFINIT"):
                log("Micromechanical force field with %i parts:&%s." % (
                    len(self.parts), ", ".join(part.name for part in self.parts)
                ))

    def add_part(self, part):
        self.parts.append(part)
        # Make the parts also accessible as simple attributes.
        name = "part_%s" % part.name
        if name in self.__dict__:
            raise ValueError("The part %s occurs twice in the micromechanical force field." % name)
        self.__dict__[name] = part

    def update_rvecs(self, rvecs):
        ForcePart.update_rvecs(self, rvecs)
        self.system.domain.update_rvecs(rvecs)

    def update_pos(self, pos):
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos

    def _internal_compute(self, gpos, vtens):
        result = sum([part.compute(gpos, vtens) for part in self.parts])
        return result



class ForcePartMechanical(ForcePart):
    
    def __init__(self, system):

        ForcePart.__init__(self, "micmec", system)
        self.system = system
        if log.do_medium:
            with log.section("FPINIT"):
                log("Force part: %s" % self.name)
                log.hline()

        # Each node in a grid has eight neighboring cells.
        # These neighbors are indexed from 0 to 7 in a fixed order.
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

        # Construct the derivatives of the neighboring cell matrices to x, y and z.
        # These derivatives are constant matrices, because the cell matrices are linear in the node coordinates.
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
    
    def _internal_compute(self, gpos, vtens):
        with timer.section("Mechanical"):
            self._compute_cell_properties()
            self._compute_gpos()
                       
            return np.sum(self.energies)

    def delta(self, i, j):
        boundary = self.system.boundary_nodes
        result = self.system.pos[j] - self.system.pos[i]
        if (i in boundary) and (j in boundary):
            result = self.system.domain.mic(result)
        return result
        
    
    def _compute_cell_properties(self):
        """Compute the cell properties."""
        # Initialize empty lists to store the cell matrices and their inverses in.
        self.cell_matrices = []
        self.inv_cell_matrices = [] 
        self.strain_tensors = []
        self.weights = []
        self.energies = []

        # Compute the deviations of the nodes from their initial, rectangular positions in the grid.
        # This is a hack to deal with periodic boundary conditions.
        
        for cell_index in range(self.system.ncells):
            
            # Store the vertices in an array.
            vertices = np.zeros((8, 3), int)
            edges = np.zeros((12, 3), float)
            for neighbor_index, node_index in self.system.surrounding_nodes[cell_index]:
                vertices[neighbor_index] = node_index
            
            edges[0] = self.delta(vertices[0], vertices[1])
            edges[1] = self.delta(vertices[2], vertices[4])
            edges[2] = self.delta(vertices[3], vertices[5])
            edges[3] = self.delta(vertices[6], vertices[7])
            
            edges[4] = self.delta(vertices[0], vertices[2])
            edges[5] = self.delta(vertices[1], vertices[4])
            edges[6] = self.delta(vertices[3], vertices[6])
            edges[7] = self.delta(vertices[5], vertices[7])
            
            edges[8] = self.delta(vertices[0], vertices[3])
            edges[9] = self.delta(vertices[2], vertices[6])
            edges[10] = self.delta(vertices[1], vertices[5])
            edges[11] = self.delta(vertices[4], vertices[7])          

            xvec = np.mean(edges[0:4], axis=0)
            yvec = np.mean(edges[4:8], axis=0)
            zvec = np.mean(edges[8:12], axis=0)
            
            # Construct the cell matrix.
            cell = np.array([xvec, yvec, zvec])

            # Initialize a strain tensor.
            strain = np.zeros((3, 3))
            cell0_inv_lst = self.system.equilibrium_inv_cell_matrices[cell_index]
            
            strain_lst = []
            for cell0_inv in cell0_inv_lst:
                strain = 0.5*((cell @ cell0_inv).T @ (cell @ cell0_inv) - np.identity(n=3))
                strain_lst.append(strain)

            # Get the elasticity tensor and free energy for each metastable state.
            elast_lst = self.system.elasticity_tensors[cell_index]
            free_energy_lst = self.system.free_energies[cell_index]

            effective_temp = self.system.effective_temps[cell_index]

            weights_cell = []
            # Iterate over all metastable states.
            for elast, free_energy, strain in zip(elast_lst, free_energy_lst, strain_lst):
                pot_energy = 0.5*cell_det*np.tensordot(strain, np.tensordot(elast, strain, axes=2), axes=2)
                weight = np.exp(-(pot_energy + free_energy)/(boltzmann*effective_temp))
                weights_cell.append(weight)
            energy_cell = -boltzmann*effective_temp*np.log(np.sum(weights_cell))

            # Store the cell matrices, the inverse cell matrices, the determinants and the strain tensors.
            self.cell_matrices.append(cell))
            self.inv_cell_matrices.append(np.linalg.inv(cell))
            self.cell_dets.append(np.linalg.det(cell))
            self.strain_tensors.append(strain_lst)
            self.energies.append(energy_cell)
            self.weights.append(weights_cell)

        return None
    
          
        
    def _compute_gpos(self):
        """
        Compute the gradient of the potential energy for each node in the network.
        The output has the same shape as the array of positions: (N, 3).
        """
        # Iterate over each node
        for node_index in range(self.system.nnodes):
            
            fx_node = 0.0
            fy_node = 0.0
            fz_node = 0.0

            # Iterate over each neighboring cell of node (k, l, m).
            for neighbor_index, cell_index in self.system.surrounding_cells[node_index]:

                cell = self.cell_matrices[cell_index]
                cell_inv = self.inv_cell_matrices[cell_index]
                cell_det = self.cell_dets[cell_index]
                
                cell0_lst = self.system.equilibrium_cell_matrices[cell_index]
                cell0_inv_lst = self.system.equilibrium_inv_cell_matrices[cell_index]
                elast_lst = self.system.elasticity_tensors[cell_index]
                
                strain_lst = self.strain_tensors[cell_index]
                weight_lst = self.weights[cell_index]
                
                weight_lst_norm = np.array(weight_lst)/np.sum(weight_lst)


                # Get the derivatives to x, y and z of the cell matrix.
                cell_xderiv = self.cell_xderivs[neighbor_index]
                cell_yderiv = self.cell_yderivs[neighbor_index]
                cell_zderiv = self.cell_zderivs[neighbor_index]
                
                cell_xtrace = np.trace(cell_inv @ cell_xderiv)
                cell_ytrace = np.trace(cell_inv @ cell_yderiv)
                cell_ztrace = np.trace(cell_inv @ cell_zderiv)
                
                fx = 0.0
                fy = 0.0
                fz = 0.0
                
                # Iterate over all stable states.
                for cell0, cell0_inv, elast, strain, weight in zip(cell0_lst, 
                                                                    cell0_inv_lst, 
                                                                    elast_lst, 
                                                                    strain_lst, 
                                                                    weight_lst_norm):                

                    # Compute the derivatives to x, y and z of the strain tensor.
                    _xmat = cell0_inv.T @ cell.T @ cell_xderiv @ cell0_inv
                    _ymat = cell0_inv.T @ cell.T @ cell_yderiv @ cell0_inv
                    _zmat = cell0_inv.T @ cell.T @ cell_zderiv @ cell0_inv
                    
                    strain_xderiv = 0.5*(_xmat.T + _xmat) # EQ.3
                    strain_yderiv = 0.5*(_ymat.T + _ymat) # EQ.3
                    strain_zderiv = 0.5*(_zmat.T + _zmat) # EQ.3
                    
                    # Compute the contribution of cell (kappa, lambda, mu) to the x component
                    # of the force acting on node (k, l, m).
                    fx_ = cell_xtrace*np.tensordot(strain.T, np.tensordot(elast, strain, axes=2), axes=2)
                    fx_ += np.tensordot(strain.T, np.tensordot(elast, strain_xderiv, axes=2), axes=2) 
                    fx_ += np.tensordot(strain_xderiv.T, np.tensordot(elast, strain, axes=2), axes=2) 
                    
                    # Compute the contribution of cell (kappa, lambda, mu) to the y component
                    # of the force acting on node (k, l, m).
                    fy_ = cell_ytrace*np.tensordot(strain.T, np.tensordot(elast, strain, axes=2), axes=2) 
                    fy_ += np.tensordot(strain.T, np.tensordot(elast, strain_yderiv, axes=2), axes=2) 
                    fy_ += np.tensordot(strain_yderiv.T, np.tensordot(elast, strain, axes=2), axes=2) 
                    
                    # Compute the contribution of cell (kappa, lambda, mu) to the y component
                    # of the force acting on node (k, l, m).
                    fz_ = cell_ztrace*np.tensordot(strain.T, np.tensordot(elast, strain, axes=2), axes=2)
                    fz_ += np.tensordot(strain.T, np.tensordot(elast, strain_zderiv, axes=2), axes=2)
                    fz_ += np.tensordot(strain_zderiv.T, np.tensordot(elast, strain, axes=2), axes=2)
                    
                    # Scale the contribution of cell (kappa, lambda, mu) according to the volume of the cell.
                    fx_ *= -0.5*cell_det 
                    fy_ *= -0.5*cell_det 
                    fz_ *= -0.5*cell_det
                
                    fx += weight*fx_
                    fy += weight*fy_
                    fz += weight*fz_
                

                # Add the contribution of cell (kappa, lambda, mu) to the total force acting on node (k, l, m).               
                fx_node += fx
                fy_node += fy
                fz_node += fz
            
            # Combine the force components into a vector.
            self.gpos[node_index, :] = -np.array([fx_node, fy_node, fz_node])
        return None
    


class ForcePartPressure(ForcePart):
    """Applies a constant istropic pressure."""
    def __init__(self, system, pext):
        """
        **ARGUMENTS**
        system
            An instance of the `System` class.
        pext
            The external pressure. (Positive will shrink the system.) In
            case of 2D-PBC, this is the surface tension. In case of 1D, this
            is the linear strain.

        This force part is only applicable to systems that are periodic.
        """
        if system.domain.nvec == 0:
            raise ValueError("The system must be periodic in order to apply a pressure")
        ForcePart.__init__(self, "press", system)
        self.system = system
        self.pext = pext
        if log.do_medium:
            with log.section("FPINIT"):
                log("Force part: %s" % self.name)
                log.hline()

    def _internal_compute(self, gpos, vtens):
        with timer.section("Pressure"):
            domain = self.system.domain
            if (vtens is not None):
                rvecs = domain.rvecs
                if domain.nvec == 1:
                    vtens += self.pext/domain.volume*np.outer(rvecs[0], rvecs[0])
                elif domain.nvec == 2:
                    vtens += self.pext/domain.volume*(
                          np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        + np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        - np.dot(rvecs[1], rvecs[0])*np.outer(rvecs[0], rvecs[1])
                        - np.dot(rvecs[0], rvecs[1])*np.outer(rvecs[1], rvecs[0])
                    )
                elif domain.nvec == 3:
                    gvecs = domain.gvecs
                    vtens += self.pext*domain.volume*np.identity(3)
                else:
                    raise NotImplementedError
            return domain.volume*self.pext


