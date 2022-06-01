#!/usr/bin/env python
# File name: mmf.py
# Description: MicroMechanicalField. The calculation of forces acting on the micromechanical nodes.
# Author: Joachim Vandewalle
# Date: 17-10-2021

"""The calculation of forces acting on the nodes of the micromechanical model."""

import numpy as np
from .nanocell import *

from molmod import boltzmann
from molmod.units import kjmol
from ..log import log, timer
from time import time

__all__ = [
    "MicMecForceField", 
    "ForcePart", 
    "ForcePartMechanical"
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

        self.temp_eff = {}
        self.efree = {}
        self.efun = {}
        self.grad_efun = {}
        for type_ in set(self.system.types):
            if int(type_) == 0:
                continue
            efun_states = []
            grad_efun_states = []
            h0 = self.system.params[f"type{int(type_)}/cell"]
            C0 = self.system.params[f"type{int(type_)}/elasticity"]
            for h0_state, C0_state in zip(h0, C0):
                efun_state = lambda state: elastic_energy(state, h0_state, C0_state)
                grad_efun_state = lambda state: grad_elastic_energy(state, h0_state, C0_state)
                efun_states.append(efun_state)
                grad_efun_states.append(grad_efun_state)
            self.efun[int(type_)] = efun_states
            self.grad_efun[int(type_)] = grad_efun_states
            self.efree[int(type_)] = self.system.params[f"type{int(type_)}/free_energy"]
            self.temp_eff[int(type_)] = self.system.params[f"type{int(type_)}/effective_temp"]

    
    def _internal_compute(self, gpos, vtens):
        with timer.section("MMFF"):
            self._compute_cell_properties()
            self._compute_gpos(gpos)
            self._compute_vtens(gpos, vtens)
                       
            return np.sum(self.epot_cells) # (3.27) and (3.36)
        
    
    def _compute_cell_properties(self):
        """Compute the cell properties."""
        self.epot_cells = []
        self.weights_cells = []
        self.gpos_cells = []
        self.verts_cells = []
        for cell_idx in range(self.system.ncells):
            type_ = int(self.system.types[cell_idx])
            # Store the nodal index of each vertex of the current cell in an array.
            verts_idxs = self.system.surrounding_nodes[cell_idx]
            # Calculate the position of each vertex.
            r0 = self.system.pos[verts_idxs[0]]
            r1 = r0 + self.delta(verts_idxs[0], verts_idxs[1])
            r2 = r0 + self.delta(verts_idxs[0], verts_idxs[2])
            r3 = r0 + self.delta(verts_idxs[0], verts_idxs[3])
            r4 = r0 + self.delta(verts_idxs[0], verts_idxs[4])
            r5 = r0 + self.delta(verts_idxs[0], verts_idxs[5])
            r6 = r0 + self.delta(verts_idxs[0], verts_idxs[6])
            r7 = r0 + self.delta(verts_idxs[0], verts_idxs[7])
            verts = np.array([r0, r1, r2, r3, r4, r5, r6, r7])
            temp_eff_ = self.temp_eff[type_]
            gpos_states = []
            epot_states = []
            # Iterate over each metastable state.
            for efree_state, efun_state, grad_efun_state in zip(self.efree[type_], 
                                                                self.efun[type_], 
                                                                self.grad_efun[type_]):
                epot_states.append(efun_state(verts) + efree_state)
                gpos_states.append(grad_efun_state(verts).reshape((8, 3)))
            epot_states = np.array(epot_states)
            epot_min = np.min(epot_states)
            weights_states = np.exp(-(epot_states - epot_min)/(boltzmann*temp_eff_)) # (3.41)
            weights_states_norm = np.array(weights_states)/np.sum(weights_states)
            epot_cell = epot_min - temp_eff_*boltzmann*np.log(np.sum(weights_states)) # (3.37)
            gpos_cell = np.zeros((8, 3))
            for weight_state, gpos_state in zip(weights_states_norm, gpos_states):
                gpos_cell += weight_state*gpos_state # (3.40)
            # Store everything.
            self.epot_cells.append(epot_cell)
            self.weights_cells.append(weights_states)
            self.gpos_cells.append(gpos_cell)
            self.verts_cells.append(verts)
            
        return None
    
    
    def _compute_gpos(self, gpos):
        """Compute the gradient of the total potential energy."""
        if gpos is None:
            return None
        # Iterate over each node
        for node_idx in range(self.system.nnodes):
            # Initialize the total force acting on the node.
            gpos_node = np.zeros(3)
            # Iterate over each surrounding cell of the node.
            for neighbor_idx, cell_idx in enumerate(self.system.surrounding_cells[node_idx]):
                if cell_idx < 0:
                    # Skip the iteration if the current cell is empty or non-existent.
                    continue
                gpos_node += self.gpos_cells[cell_idx][neighbor_idx]
            gpos[node_idx, :] = gpos_node
        return None

    
    def _compute_vtens(self, gpos, vtens):
        if (vtens is None) or (gpos is None):
            return None
        vtens[:] = np.einsum("ijk,ijl->kl", self.gpos_cells, self.verts_cells) # (4.1)
        return None

    
    def delta(self, i, j):
        boundary = self.system.boundary_nodes
        dvec = self.system.pos[j] - self.system.pos[i]
        if (i in boundary) and (j in boundary):
            self.system.domain.mic(dvec)
        return dvec



