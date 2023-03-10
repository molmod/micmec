#!/usr/bin/env python

#   MicMec 1.0, the first implementation of the micromechanical model, ever.
#               Copyright (C) 2022  Joachim Vandewalle
#                    joachim.vandewalle@hotmail.be
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, either version 3 of the License, or
#                  (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#              GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see https://www.gnu.org/licenses/.


"""MicMecForceField, the micromechanical force field."""

import numpy as np

from functools import partial

from micmec.pes.nanocell_original import (
    elastic_energy_nanocell,
    grad_elastic_energy_nanocell,
)
from micmec.utils import Grid

from molmod import boltzmann
from micmec.log import log, timer

__all__ = ["MicMecForceField", "ForcePart", "ForcePartMechanical"]


class ForcePart(object):
    """Base class for anything that can compute energies (and optionally gradient and virial) for a ``System`` object,
    as part of a larger micromechanical force field (MMFF) model.

    Parameters
    ----------
    name : str
        A name for this part of the micromechanical force field (MMFF).
        This name must adhere to the following conventions: all lower case, no white space, and short.
        It is used to construct part_* attributes in the MicMecForceField class, where * is the name.
    system : micmec.system.System
        The system to which this part of the MMFF applies.
    """

    def __init__(self, name, system):
        self.name = name
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.nnodes, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.clear()

    def clear(self):
        """Fill in ``nan`` values in the cached results to indicate that they have become invalid."""
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):
        """Let the ``ForcePart`` object know that the domain vectors have changed.

        Parameters
        ----------
        rvecs : numpy.ndarray, shape=(``nper``, 3)
            The new domain vectors.
        """
        self.clear()

    def update_pos(self, pos):
        """Let the ``ForcePart`` object know that the nodal positions have changed.

        Parameters
        ----------
        pos : numpy.ndarray, shape=(``nnodes``, 3)
            The new nodal coordinates.
        """
        self.clear()

    def compute(self, gpos=None, vtens=None):
        """Compute the energy of this part of the MMFF.

        The only variable inputs for the compute routine are the nodal positions and the domain vectors, which can be
        changed through the ``update_rvecs`` and ``update_pos`` methods.
        All other aspects of the MMFF are considered to be fixed between subsequent compute calls.
        If changes other than positions or domain vectors are needed, one must construct a new MMFF instance.

        Parameters
        ----------
        gpos : numpy.ndarray, shape=(``nnodes``, 3), optional
            The derivatives of the energy towards the Cartesian coordinates of the nodes.
            ("g" stands for gradient and "pos" for positions.)
        vtens : numpy.ndarray, shape=(3, 3), optional
            The force contribution to the pressure tensor, also known as the virial tensor.
            It represents the derivative of the energy towards uniform deformations, including changes in the shape
            of the unit domain.
            ("v" stands for virial and "tens" stands for tensor.)

        Raises
        ------
        ValueError
            If the energy is not-a-number (``nan``) or if the ``gpos`` or ``vtens`` array contains a ``nan``.

        Returns
        -------
        energy : float
            The (potential) energy (of the MMFF).

        Notes
        -----
        The optional arguments are Fortran-style output arguments.
        When they are present, the corresponding results are computed and **added** to the current contents of the array.
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
            raise ValueError("The energy is not-a-number (``nan``).")
        if gpos is not None:
            if np.isnan(mmf_gpos).any():
                raise ValueError(
                    "Some ``gpos`` element(s) is/are not-a-number (``nan``)."
                )
            gpos += mmf_gpos
        if vtens is not None:
            if np.isnan(mmf_vtens).any():
                raise ValueError(
                    "Some ``vtens`` element(s) is/are not-a-number (``nan``)."
                )
            vtens += mmf_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        """Subclasses implement their compute code here."""
        raise NotImplementedError


class MicMecForceField(ForcePart):
    """A complete micromechanical force field (MMFF) model.

    Parameters
    ----------
    system : micmec.system.System
        The micromechanical system.
    parts : list of micmec.pes.mmff.ForcePart
        The different types of contributions to the MMFF.
    """

    def __init__(self, system, parts):
        ForcePart.__init__(self, "all", system)
        self.system = system
        self.parts = []
        for part in parts:
            self.add_part(part)
        if log.do_medium:
            with log.section("FFINIT"):
                log(
                    "Force field with %i parts:&%s."
                    % (len(self.parts), ", ".join(part.name for part in self.parts))
                )

    def add_part(self, part):
        """Add a ``ForcePart`` object to the MMFF."""
        self.parts.append(part)
        # Make the parts also accessible as simple attributes.
        name = "part_%s" % part.name
        if name in self.__dict__:
            raise ValueError(
                "The part %s occurs twice in the micromechanical force field." % name
            )
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
    """The micromechanical part of the MMFF."""

    def __init__(self, system):
        ForcePart.__init__(self, "micmec", system)
        self.system = system
        if log.do_medium:
            with log.section("FPINIT"):
                log("Force part: %s" % self.name)
                log.hline()

        self.pbc = self.get_pbc(self.system.domain.rvecs)
        self.mic = self.get_mic(self.system.grid, self.pbc)

        # Filter such that only scalar parameters remain.
        params1 = {}
        params2 = {}
        params3 = {}
        params4 = {}
        for key, item in self.system.params.items():
            if "effective_temp" in key:
                params1[int(key.split("/")[0][4:])] = item
            if "free_energy" in key:
                params2[int(key.split("/")[0][4:])] = item
            if "cell" in key:
                params3[int(key.split("/")[0][4:])] = item
            if "elasticity" in key:
                params4[int(key.split("/")[0][4:])] = item

        self.efun = partial(
            deformation,
            mic=self.mic,
            types=self.system.types,
            surrounding_nodes=self.system.surrounding_nodes,
            params1=params1,
            params2=params2,
            params3=params3,
            params4=params4,
        )
        self.epot_cells = None
        self.gpos_cells = None
        self.verts_cells = None

    @staticmethod
    def get_pbc(rvecs):
        nper = rvecs.shape[0]
        pbc = nper > 0
        if pbc and nper != 3:
            raise ValueError(
                "Attribute `rvecs` only supports finite systems or 3D periodic systems, "
                f"not {nper}D periodic systems."
            )
        return pbc

    @staticmethod
    def get_mic(grid, pbc=True):
        grid_structure = Grid(grid, pbc=pbc)
        mic = np.zeros((grid_structure.nnodes, grid_structure.nnodes, 3))
        if not pbc:
            return mic
        sign = lambda x: 2 * (x > 0) - 1
        boundary = grid_structure.boundary_nodes
        nodes = np.array(grid_structure.nodes)
        kij_max = grid_structure.nx_nodes - 1
        lij_max = grid_structure.ny_nodes - 1
        mij_max = grid_structure.nz_nodes - 1
        for j in range(grid_structure.nnodes):
            for i in range(j):
                if (i not in boundary) or (j not in boundary):
                    continue
                kij = nodes[j, 0] - nodes[i, 0]
                lij = nodes[j, 1] - nodes[i, 1]
                mij = nodes[j, 2] - nodes[i, 2]
                if abs(kij) == kij_max:
                    mic[i, j, 0] = -sign(kij) if kij_max != 1 else (kij == -1)
                    mic[j, i, 0] = sign(kij) if kij_max != 1 else (kij == 1)
                if abs(lij) == lij_max:
                    mic[i, j, 1] = -sign(lij) if lij_max != 1 else (lij == -1)
                    mic[j, i, 1] = sign(lij) if lij_max != 1 else (lij == 1)
                if abs(mij) == mij_max:
                    mic[i, j, 2] = -sign(mij) if mij_max != 1 else (mij == -1)
                    mic[j, i, 2] = sign(mij) if mij_max != 1 else (mij == 1)
        return mic

    def _internal_compute(self, gpos, vtens):
        with timer.section("MMFF"):
            self.epot_cells, self.gpos_cells, self.verts_cells = self.efun(
                self.system.pos, self.system.domain.rvecs
            )
            if gpos is not None:
                self._compute_gpos(gpos)
            if vtens is not None:
                self._compute_vtens(vtens)
            return self._compute_epot()  # (3.27) and (3.36)

    def _compute_epot(self):
        """Compute the potential energy (of the MMFF)."""
        return np.sum(self.epot_cells)

    def _compute_gpos(self, gpos):
        """Compute the gradient of the potential energy (of the MMFF)."""
        # Iterate over each node
        for node_idx in range(self.system.nnodes):
            # Initialize the total force acting on the node.
            gpos_node = np.zeros(3)
            # Iterate over each surrounding cell of the node.
            for neighbor_idx, cell_idx in enumerate(
                self.system.surrounding_cells[node_idx]
            ):
                if cell_idx < 0:
                    # Skip the iteration if the current cell is empty or non-existent.
                    continue
                gpos_node += self.gpos_cells[cell_idx][neighbor_idx]
            gpos[node_idx, :] = gpos_node
        return None

    def _compute_vtens(self, vtens):
        """Compute the virial tensor of the simulation domain."""
        vtens[:] = np.einsum("ijk,ijl->kl", self.gpos_cells, self.verts_cells)  # (4.1)
        return None


def deformation(
    pos, rvecs, mic, types, surrounding_nodes, params1, params2, params3, params4
):
    """Compute the deformation properties: the instantaneous potential energy, the gradient and vertices of
    each cell, taking into account the minimum image convention.
    """
    ncells = len(types)

    def delta(i, j):
        """Compute the difference vector between node i and node j, taking into account the minimum image convention.

        Parameters
        ----------
        i, j : int
            The indices of the two micromechanical nodes in question.

        Returns
        -------
        dvec : numpy.ndarray, shape=(3,)
            The difference vector between node i and node j.
        """
        dvec = pos[j] - pos[i]
        dvec += (
            rvecs[0] * mic[i, j, 0] + rvecs[1] * mic[i, j, 1] + rvecs[2] * mic[i, j, 2]
        )
        return dvec

    epot_cells = []
    gpos_cells = []
    verts_cells = []
    for cell_idx in range(ncells):
        type_ = types[cell_idx]

        # Store the nodal index of each vertex of the current cell in an array.
        verts_idxs = surrounding_nodes[cell_idx]

        # Calculate the posrvecs[0] * mic[i, j, 0] + rvecs[1] * mic[i, j, 1] + rvecs[2] * mic[i, j, 2]ition of each vertex.
        r0 = pos[verts_idxs[0]]
        r1 = r0 + delta(verts_idxs[0], verts_idxs[1])
        r2 = r0 + delta(verts_idxs[0], verts_idxs[2])
        r3 = r0 + delta(verts_idxs[0], verts_idxs[3])
        r4 = r0 + delta(verts_idxs[0], verts_idxs[4])
        r5 = r0 + delta(verts_idxs[0], verts_idxs[5])
        r6 = r0 + delta(verts_idxs[0], verts_idxs[6])
        r7 = r0 + delta(verts_idxs[0], verts_idxs[7])
        verts_cell = np.array([r0, r1, r2, r3, r4, r5, r6, r7])

        epot_states = []
        gpos_states = []
        # Iterate over each metastable state.
        temp_eff = params1[int(type_)]
        for efree_state, h0_state, C0_state in zip(
            params2[int(type_)], params3[int(type_)], params4[int(type_)]
        ):
            epot_states.append(
                elastic_energy_nanocell(verts_cell, h0_state, C0_state) + efree_state
            )
            gpos_states.append(
                grad_elastic_energy_nanocell(verts_cell, h0_state, C0_state)
            )
        epot_states = np.array(epot_states)
        gpos_states = np.array(gpos_states)
        epot_min = np.min(epot_states)
        weights_states = np.exp(
            -(epot_states - epot_min) / (boltzmann * temp_eff)
        )  # (3.41)
        weights_states_norm = np.array(weights_states) / np.sum(weights_states)
        epot_cell = epot_min - temp_eff * boltzmann * np.log(
            np.sum(weights_states)
        )  # (3.37)
        gpos_cell = np.zeros((8, 3))
        for weight_state, gpos_state in zip(weights_states_norm, gpos_states):
            gpos_cell += weight_state * gpos_state  # (3.40)
        epot_cells.append(epot_cell)
        gpos_cells.append(gpos_cell)
        verts_cells.append(verts_cell)

    return np.array(epot_cells), np.array(gpos_cells), np.array(verts_cells)
