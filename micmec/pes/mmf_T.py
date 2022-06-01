#!/usr/bin/env python
# File name: mmf.py
# Description: MicroMechanicalField. The calculation of forces acting on the micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

"""The calculation of forces acting on the nodes of the micromechanical model."""

import numpy as np

from molmod import boltzmann

from ..log import log, timer

from time import time

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
            self._compute_vtens(gpos, vtens)
                       
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
        self.cell_verts = []
        
        for cell_idx in range(self.system.ncells):
            # Store the nodal index of each vertex of the current cell in an array.
            vertices = np.zeros((8,), int)
            for neighbor_idx, node_idx in enumerate(self.system.surrounding_nodes[cell_idx]):
                vertices[neighbor_idx] = node_idx
            # Calculate the position of each vertex.
            r0 = self.system.pos[vertices[0]]
            r1 = r0 + self.delta(vertices[0], vertices[1])
            r2 = r0 + self.delta(vertices[0], vertices[2])
            r3 = r0 + self.delta(vertices[0], vertices[3])
            r4 = r0 + self.delta(vertices[0], vertices[4])
            r5 = r0 + self.delta(vertices[0], vertices[5])
            r6 = r0 + self.delta(vertices[0], vertices[6])
            r7 = r0 + self.delta(vertices[0], vertices[7])
            # Store each edge vector of the current cell in an array.
            edges = np.zeros((12, 3), float)
            # Edges pointing in the x-direction.
            edges[0] = r1 - r0
            edges[1] = r4 - r2
            edges[2] = r5 - r3
            edges[3] = r7 - r6
            # Edges pointing in the y-direction.            
            edges[4] = r2 - r0
            edges[5] = r4 - r1
            edges[6] = r6 - r3
            edges[7] = r7 - r5
            # Edges pointing in the z-direction.
            edges[8] = r3 - r0
            edges[9] = r6 - r2
            edges[10] = r5 - r1
            edges[11] = r7 - r4
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
            cell_mat = np.array([h0, h1, h2, h3, h4, h5, h6, h7])
            cell_det = np.linalg.det(cell_mat)
            cell_inv = np.linalg.inv(cell_mat)
            cell_vert = np.array([r0, r1, r2, r3, r4, r5, r6, r7])
            # Load the equilibrium cell properties.
            # Each metastable state has a different equilibrium inverse cell matrix, elasticity tensor
            # and free energy. The effective temperature is an additional fitting parameter.
            cell0_eff_temp = self.system.effective_temps[cell_idx]
            # Initialize the list of strain tensors and potential energies for each metastable state.
            cell_strain_lst = []
            cell_epot_lst = []
            # Iterate over each metastable state.
            for cell0_inv, cell0_elast, cell0_efree in zip(self.system.equilibrium_inv_cell_matrices[cell_idx], 
                                                            self.system.elasticity_tensors[cell_idx], 
                                                            self.system.free_energies[cell_idx]):
                cell_strain = self.strain(cell_mat, cell0_inv)
                cell_epot = self.elastic_energy(cell_strain, cell0_elast, cell_det)
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
            self.cell_verts.append(cell_vert)
        return None
    
    def _compute_gpos(self, gpos):
        """
        Compute the gradient of the potential energy for each node in the network.
        The output has the same shape as the array of positions: (N, 3).
        """
        if gpos is None:
            return None
        self.cell_gpos_contribs = np.zeros((self.system.ncells, 8, 3))
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
                # Load the equilibrium properties.
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
                for cell0_inv, cell0_elast, cell_strain, cell_weight in zip(cell0_inv_lst, 
                                                                            cell0_elast_lst, 
                                                                            cell_strain_lst, 
                                                                            cell_weight_lst_norm):              
                    cell_xderiv = self.cell_xderivs[neighbor_idx]
                    cell_yderiv = self.cell_yderivs[neighbor_idx]
                    cell_zderiv = self.cell_zderivs[neighbor_idx]
                    # Calculate the force contribution due to every representation and
                    # multiply with the appropriate thermodynamic weight for each metastable state.
                    f += cell_weight*self.force(cell_mat, 
                                                cell_xderiv, 
                                                cell_yderiv, 
                                                cell_zderiv, 
                                                cell0_inv, 
                                                cell0_elast, 
                                                cell_strain, 
                                                cell_det, 
                                                cell_inv)
                # Store the f contribution to calculate the virial tensor for later.
                self.cell_gpos_contribs[cell_idx][neighbor_idx] -= f
                # Add the contribution of the current cell to the total force acting on the node.               
                ftot += f
            gpos[node_idx, :] = -ftot
        return None

    def _compute_vtens(self, gpos, vtens):
        if (vtens is None) or (gpos is None):
            return None
        vtens_ = np.zeros((3, 3))
        vtens_ += np.einsum("ijk,ijl->kl", self.cell_verts, self.cell_gpos_contribs)
        vtens[:] = vtens_
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
        return 0.0625*np.einsum("a,aij,ijkl,akl", h_det, eps, C0, eps)

    @staticmethod
    def strain(h, h0_inv):
        """
        Mechanical strain.  
        h   
            cell matrix
        h0_inv
            inverse equilibrium cell matrix
        """
        mat = np.einsum("...ij,jk->...ik", h, h0_inv)
        iden = np.array([np.identity(3) for _ in range(8)])
        return 0.5*(np.einsum("...ji,...jk->...ik", mat, mat) - iden)
    
    @staticmethod
    def force(h, h_xderiv, h_yderiv, h_zderiv, h0_inv, C0, eps=None, h_det=None, h_inv=None):
        #    8x3x3,8x3x3,    8x3x3,    8x3x3,    3x3,  3x3x3x3, 8x3x3, 8,          8x3x3
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
        f = np.zeros((3, 8))
        if eps is None:
            eps = strain(h, h0_inv)
        if h_det is None:
            h_det = np.linalg.det(h)
        if h_inv is None:
            h_inv = np.linalg.inv(h)
        
        h_xtrace = np.einsum("...ij,...ji", h_inv, h_xderiv) #8
        h_ytrace = np.einsum("...ij,...ji", h_inv, h_yderiv)
        h_ztrace = np.einsum("...ij,...ji", h_inv, h_zderiv)

        mat = np.einsum("...ij,jk->...ki", h, h0_inv)
        xmat = np.einsum("...ij,...jk->...ik", mat, np.einsum("...ij,jk->...ik", h_xderiv, h0_inv)) #8x3x3
        ymat = np.einsum("...ij,...jk->...ik", mat, np.einsum("...ij,jk->...ik", h_yderiv, h0_inv))
        zmat = np.einsum("...ij,...jk->...ik", mat, np.einsum("...ij,jk->...ik", h_zderiv, h0_inv))
        
        eps_xderiv = 0.5*(np.einsum("...ji", xmat) + xmat)
        eps_yderiv = 0.5*(np.einsum("...ji", ymat) + ymat)
        eps_zderiv = 0.5*(np.einsum("...ji", zmat) + zmat)

        stress = np.einsum("ijkl,...kl->...ij", C0, eps) #8x3x3
        quad_form = np.einsum("...ij,...ij", eps, stress) #8
        
        # Compute the contribution of cell (kappa, lambda, mu) to the x component
        # of the force acting on node (k, l, m).
        xterm = np.einsum("...ji,...ij", eps_xderiv, stress) #8
        f[0] += h_xtrace*quad_form
        f[0] += 2.0*xterm
        
        # Compute the contribution of cell (kappa, lambda, mu) to the y component
        # of the force acting on node (k, l, m).
        yterm = np.einsum("...ji,...ij", eps_yderiv, stress)
        f[1] += h_ytrace*quad_form
        f[1] += 2.0*yterm
        
        # Compute the contribution of cell (kappa, lambda, mu) to the y component
        # of the force acting on node (k, l, m).
        zterm = np.einsum("...ji,...ij", eps_zderiv, stress)
        f[2] += h_ztrace*quad_form
        f[2] += 2.0*zterm
        
        # Scale the contribution of cell (kappa, lambda, mu) according to the volume of the cell.
        return -0.0625*np.einsum("i,...i", h_det, f)


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


