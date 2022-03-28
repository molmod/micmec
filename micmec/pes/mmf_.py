#!/usr/bin/env python
# File name: mmf.py
# Description: MicroMechanicalField. The calculation of forces acting on the micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

"""The calculation of forces acting on the nodes of the micromechanical model."""

import numpy as np

from molmod import boltzmann

from ..log import log, timer

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
        self.gpos = np.zeros((system.nnodes, 3), float)
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
            mmf_gpos = None
        else:
            mmf_gpos = self.gpos
            mmf_gpos[:] = 0.0
        
        if vtens is None:
            mmf_vtens = None
        else:
            mmf_vtens = self.vtens
            mmf_vtens[:] = 0.0
        
        self.energy = self._internal_compute(mmf_gpos, mmf_vtens)
        
        if np.isnan(self.energy):
            raise ValueError("The energy is not-a-number (nan).")
        if gpos is not None:
            if np.isnan(mmf_gpos).any():
                raise ValueError("Some gpos element(s) is/are not-a-number (nan).")
            gpos += mmf_gpos
        if vtens is not None:
            if np.isnan(mmf_vtens).any():
                raise ValueError("Some vtens element(s) is/are not-a-number (nan).")
            vtens += mmf_vtens
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
            with log.section("FFINIT"):
                log("Force field with %i parts:&%s." % (
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
            np.array(( 0, 0, 0)),
            np.array((-1, 0, 0)),
            np.array(( 0,-1, 0)),
            np.array(( 0, 0,-1)),
            np.array((-1,-1, 0)),
            np.array((-1, 0,-1)),
            np.array(( 0,-1,-1)),
            np.array((-1,-1,-1))
        ]

        # Initialize the derivatives of the neighboring cell matrices to x, y and z.
        self.cell_xderivs = []
        self.cell_yderivs = []
        self.cell_zderivs = []

        # Construct the derivatives of the neighboring cell matrices to x, y and z.
        # These derivatives are constant matrices, because the cell matrices are linear in the node coordinates.
        for neighbor_cell in neighbor_cells:
            xderivs = []
            yderivs = []
            zderivs = []
            
            for neighbor_cell_ in neighbor_cells:
                
                xderiv = np.zeros((3, 3))
                yderiv = np.zeros((3, 3))
                zderiv = np.zeros((3, 3))

                deriv = np.where(neighbor_cell == -1.0, 1.0, -1.0)
                deriv = np.array([1.0 if n == -1 else -1.0 for n in neighbor_cell])
                
                dist_vec = np.abs(neighbor_cell - neighbor_cell_)
                dist = np.sum(dist_vec)           

                if dist == 0.0:
                    xderiv[:, 0] = deriv
                    yderiv[:, 1] = deriv
                    zderiv[:, 2] = deriv
                elif dist == 1.0:
                    xderiv[dist_vec == 1.0, 0] = deriv[dist_vec == 1.0]
                    yderiv[dist_vec == 1.0, 1] = deriv[dist_vec == 1.0]
                    zderiv[dist_vec == 1.0, 2] = deriv[dist_vec == 1.0]
                else:
                    pass

                xderivs.append(xderiv)
                yderivs.append(yderiv)
                zderivs.append(zderiv)
            
            self.cell_xderivs.append(xderivs)
            self.cell_yderivs.append(yderivs)
            self.cell_zderivs.append(zderivs)
    
    def _internal_compute(self, gpos, vtens):
        with timer.section("MMFF"):
            self._compute_cell_properties()
            self._compute_gpos(gpos)
            self._compute_vtens(vtens)
                       
            return np.sum(self.cell_epots)
        
    
    def _compute_cell_properties(self):
        """Compute the cell properties."""
        # Initialize an empty list to store the cell matrices and their inverses in.
        self.cell_mats = []
        self.cell_invs = []
        self.cell_dets = []
        self.cell_strains = []
        self.cell_epots = []
        self.cell_weights = []
        
        for cell_idx in range(self.system.ncells):
            # Store the nodal index of each vertex of the current cell in an array.
            vertices = np.zeros((8,), int)
            for neighbor_idx, node_idx in enumerate(self.system.surrounding_nodes[cell_idx]):
                vertices[neighbor_idx] = node_idx
            # Store each edge vector of the current cell in an array.
            edges = np.zeros((12, 3), float)
            # Edges pointing in the x-direction.
            edges[0] = self.delta(vertices[0], vertices[1])
            edges[1] = self.delta(vertices[2], vertices[4])
            edges[2] = self.delta(vertices[3], vertices[5])
            edges[3] = self.delta(vertices[6], vertices[7])
            # Edges pointing in the y-direction.            
            edges[4] = self.delta(vertices[0], vertices[2])
            edges[5] = self.delta(vertices[1], vertices[4])
            edges[6] = self.delta(vertices[3], vertices[6])
            edges[7] = self.delta(vertices[5], vertices[7])
            # Edges pointing in the z-direction.
            edges[8] = self.delta(vertices[0], vertices[3])
            edges[9] = self.delta(vertices[2], vertices[6])
            edges[10] = self.delta(vertices[1], vertices[5])
            edges[11] = self.delta(vertices[4], vertices[7])
            # Construct each possible cell matrix of the current cell.
            h0 = np.array([edges[0], edges[4], edges[8]])
            h1 = np.array([edges[0], edges[5], edges[10]])
            h2 = np.array([edges[1], edges[4], edges[9]])
            h3 = np.array([edges[2], edges[6], edges[8]])
            h4 = np.array([edges[1], edges[5], edges[11]])
            h5 = np.array([edges[2], edges[7], edges[10]])
            h6 = np.array([edges[3], edges[6], edges[9]])
            h7 = np.array([edges[3], edges[7], edges[11]])
            # Construct the cell properties:
            # the cell matrix, the determinant of the cell matrix (i.e. the cell vollume)
            # and the inverse cell matrix.
            cell_mat = [h0, h1, h2, h3, h4, h5, h6, h7]
            cell_det = [np.linalg.det(h) for h in cell_mat]
            cell_inv = [np.linalg.inv(h) for h in cell_mat]
            # Load the equilibrium cell properties.
            # Each metastable state has a different equilibrium inverse cell matrix, elasticity tensor
            # and free energy. The effective temperature is an additional fitting parameter.
            cell0_inv_lst = self.system.equilibrium_inv_cell_matrices[cell_idx]
            cell0_elast_lst = self.system.elasticity_tensors[cell_idx]
            cell0_efree_lst = self.system.free_energies[cell_idx]
            cell0_eff_temp = self.system.effective_temps[cell_idx]
            # Initialize the list of strain tensors and potential energies for each metastable state.
            cell_strain_lst = []
            cell_epot_lst = []
            # Iterate over each metastable state.
            for h0_inv, C0, cell0_efree in zip(cell0_inv_lst, cell0_elast_lst, cell0_efree_lst):
                cell_strain = [self.strain(h, h0_inv) for h in cell_mat]
                cell_epot = 0.125*np.sum([self.elastic_energy(eps, C0, h_det) for eps, h_det in zip(cell_strain, cell_det)])
                cell_strain_lst.append(cell_strain)
                cell_epot_lst.append(cell_epot + cell0_efree)
            cell_epot_lst = np.array(cell_epot_lst)
            cell_epot_min = np.min(cell_epot_lst)
            cell_weight_lst = np.exp(-(cell_epot_lst - cell_epot_min)/(boltzmann*cell0_eff_temp))
            cell_epot = cell_epot_min - cell0_eff_temp*boltzmann*np.log(np.sum(cell_weight_lst))

            # Store the cell matrices, the inverse cell matrices, the determinants and the strain tensors.
            self.cell_mats.append(cell_mat)
            self.cell_invs.append(cell_inv)
            self.cell_dets.append(cell_det)
            self.cell_strains.append(cell_strain_lst)
            self.cell_epots.append(cell_epot)
            self.cell_weights.append(cell_weight_lst)

        return None
    
    def _compute_gpos(self, gpos):
        """
        Compute the gradient of the potential energy for each node in the network.
        The output has the same shape as the array of positions: (N, 3).
        """
        if gpos is None:
            return None
        # Iterate over each node
        for node_idx in range(self.system.nnodes):
            # Initialize the total force acting on the node.
            ftot = np.zeros(3)
            # Iterate over each surrounding cell of the node.
            for neighbor_idx, cell_idx in enumerate(self.system.surrounding_cells[node_idx]):
                if cell_idx < 0:
                    # Skip the iteration if the current cell is empty or non-existent.
                    # An empty or non-existent cell does not contribute to the force acting on the node.
                    continue
                # Load the 
                cell0_inv_lst = self.system.equilibrium_inv_cell_matrices[cell_idx]
                cell0_elast_lst = self.system.elasticity_tensors[cell_idx]
                # Load the precalculated properties of the cell:
                # the cell matrix, the inverse cell matrix, the determinant of the cell matrix (i.e. the cell volume),
                # the strain tensors for each metastable state and the thermodynamic weights for each metastable state.
                cell_mat = self.cell_mats[cell_idx]
                cell_inv = self.cell_invs[cell_idx]
                cell_det = self.cell_dets[cell_idx]
                cell_strain_lst = self.cell_strains[cell_idx]
                cell_weight_lst = self.cell_weights[cell_idx]
                # Normalize the thermodynamic weights.
                cell_weight_lst_norm = np.array(cell_weight_lst)/np.sum(cell_weight_lst)
                # Initialize the force contribution acting on the node due to the deformation of the current cell.
                f = np.zeros(3)
                # Iterate over each metastable state of the current cell.
                for h0_inv, C0, cell_strain, cell_weight in zip(cell0_inv_lst, cell0_elast_lst, cell_strain_lst, cell_weight_lst_norm):              
                    f_ = np.zeros(3)
                    # Iterate over each possible cell matrix of the current cell.
                    for idx, (h, h_det, h_inv, eps) in enumerate(zip(cell_mat, cell_det, cell_inv, cell_strain)):
                        # Get the derivatives to x, y and z of the current cell matrix.
                        h_xderiv = self.cell_xderivs[neighbor_idx][idx]
                        h_yderiv = self.cell_yderivs[neighbor_idx][idx]
                        h_zderiv = self.cell_zderivs[neighbor_idx][idx]
                        # Calculate the force contribution due to the current cell matrix.
                        f_ += 0.125*self.force(h, h_xderiv, h_yderiv, h_zderiv, h0_inv, C0, eps, h_det, h_inv)
                    # Multiply with the appropriate thermodynamic weight for each metastable state.
                    f += cell_weight*f_
                # Add the contribution of the current cell to the total force acting on the node.               
                ftot += f
            gpos[node_idx, :] = -ftot
        return None

    def _compute_vtens(self, vtens):
        if vtens is not None:
            vtens[:] = np.dot(self.system.pos.T, self.gpos)
        return None

    
    def delta(self, i, j):
        boundary = self.system.boundary_nodes
        dvec = self.system.pos[j] - self.system.pos[i]
        if (i in boundary) and (j in boundary):
            self.system.domain.mic(dvec)
        return dvec

    @staticmethod
    def elastic_energy(eps, C0, h_det):
        """
        Elastic deformation energy in the harmonic approximation.
        eps
            strain tensor
        C0
            elasticity tensor 
        h_det
            determinant of cell matrix (volume of cell)
        """
        return 0.5*h_det*np.tensordot(eps, np.tensordot(C0, eps, axes=2), axes=2)

    @staticmethod
    def strain(h, h0_inv):
        """
        Mechanical strain.  
        h   
            cell matrix
        h0_inv
            inverse equilibrium cell matrix
        """
        mat = h @ h0_inv
        iden = np.identity(3)
        return 0.5*(mat.T @ mat - iden)
    
    @staticmethod
    def force(h, h_xderiv, h_yderiv, h_zderiv, h0_inv, C0, eps=None, h_det=None, h_inv=None):
        """
        Micromechanical force.
        h
            cell matrix
        h_xderiv, h_yderiv, h_zderiv
            partial derivatives of cell matrix to the cartesian coordinates of a node
        h0_inv
            inverse equilibrium cell matrix
        C0
            (equilibrium) elasticity tensor
        eps
            strain tensor
        h_det
            determinant of cell matrix (volume of cell)
        h_inv
            inverse cell matrix
        """
        if eps is None:
            eps = strain(h, h0_inv)
        if h_det is None:
            h_det = np.linalg.det(h)
        if h_inv is None:
            h_inv = np.linalg.inv(h)
        
        h_xtrace = np.trace(h_inv @ h_xderiv)        
        h_ytrace = np.trace(h_inv @ h_yderiv)
        h_ztrace = np.trace(h_inv @ h_zderiv)

        xmat = h0_inv.T @ h.T @ h_xderiv @ h0_inv
        ymat = h0_inv.T @ h.T @ h_yderiv @ h0_inv
        zmat = h0_inv.T @ h.T @ h_zderiv @ h0_inv
        
        eps_xderiv = 0.5*(xmat.T + xmat)
        eps_yderiv = 0.5*(ymat.T + ymat)
        eps_zderiv = 0.5*(zmat.T + zmat)
        
        # Compute the contribution of cell (kappa, lambda, mu) to the x component
        # of the force acting on node (k, l, m).
        fx = h_xtrace*np.tensordot(eps.T, np.tensordot(C0, eps, axes=2), axes=2)
        fx += np.tensordot(eps.T, np.tensordot(C0, eps_xderiv, axes=2), axes=2) 
        fx += np.tensordot(eps_xderiv.T, np.tensordot(C0, eps, axes=2), axes=2) 
        
        # Compute the contribution of cell (kappa, lambda, mu) to the y component
        # of the force acting on node (k, l, m).
        fy = h_ytrace*np.tensordot(eps.T, np.tensordot(C0, eps, axes=2), axes=2) 
        fy += np.tensordot(eps.T, np.tensordot(C0, eps_yderiv, axes=2), axes=2) 
        fy += np.tensordot(eps_yderiv.T, np.tensordot(C0, eps, axes=2), axes=2) 
        
        # Compute the contribution of cell (kappa, lambda, mu) to the y component
        # of the force acting on node (k, l, m).
        fz = h_ztrace*np.tensordot(eps.T, np.tensordot(C0, eps, axes=2), axes=2)
        fz += np.tensordot(eps.T, np.tensordot(C0, eps_zderiv, axes=2), axes=2)
        fz += np.tensordot(eps_zderiv.T, np.tensordot(C0, eps, axes=2), axes=2)
        
        # Scale the contribution of cell (kappa, lambda, mu) according to the volume of the cell.
        fx *= -0.5*h_det 
        fy *= -0.5*h_det 
        fz *= -0.5*h_det

        return np.array([fx, fy, fz])



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


