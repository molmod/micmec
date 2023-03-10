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


"""MicMecForceField, the micromechanical force field.

This is the JAX-enabled version of the default micromechanical force field. It uses automatic differentiation
and Just-In-Time compilation. Despite these benefits, there is no guarantee this code is efficient or fast.
"""

import jax

import numpy as np
import jax.numpy as jnp

from functools import partial

from molmod import boltzmann
from micmec.log import log, timer
from micmec.utils import Grid

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
        self.mic = self.get_mic(self.system.grid)

        # Filter such that only scalar parameters remain (JAX requirement).
        self.params1 = {}
        self.params2 = {}
        self.params3 = {}
        self.params4 = {}
        for key, item in self.system.params.items():
            if "effective_temp" in key:
                self.params1[int(key.split("/")[0][4:])] = item
            if "free_energy" in key:
                self.params2[int(key.split("/")[0][4:])] = item
            if "cell" in key:
                self.params3[int(key.split("/")[0][4:])] = item
            if "elasticity" in key:
                self.params4[int(key.split("/")[0][4:])] = item

        # Construct potential energy function.
        self.fn_epot = jax.jit(
            partial(
                elastic_energy,
                mic=self.mic,
                types=self.system.types,
                surrounding_nodes=self.system.surrounding_nodes,
                params1=self.params1,
                params2=self.params2,
                params3=self.params3,
                params4=self.params4,
            )
        )
        # Perform automatic differentiation.
        self.fn_gpos = jax.jit(jax.grad(self.fn_epot, argnums=(0,)))
        self.fn_vtens = jax.jit(jax.grad(self.fn_epot, argnums=(2,)))

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
            epot = np.array(
                self.fn_epot(self.system.pos, self.system.domain.rvecs, np.identity(3))
            )
            if gpos is not None:
                gpos[:] = np.array(
                    self.fn_gpos(
                        self.system.pos, self.system.domain.rvecs, np.identity(3)
                    )[0]
                )
            if vtens is not None:
                # vtens[:] = np.array(self.system.domain.rvecs.T @ self.fn_vtens(self.system.pos, self.system.domain.rvecs)[0])
                vtens[:] = np.array(
                    self.fn_vtens(
                        self.system.pos, self.system.domain.rvecs, np.identity(3)
                    )[0]
                )
            return epot  # (3.27) and (3.36)


def elastic_energy(
    pos,
    rvecs,
    deformation,
    mic,
    types,
    surrounding_nodes,
    params1,
    params2,
    params3,
    params4,
):
    """Compute the instantaneous potential energy."""
    pos = jnp.dot(pos, deformation)
    rvecs = jnp.dot(rvecs, deformation)
    epot = 0.0
    ncells = len(types)

    # Construct a multiplicator array.
    # This array converts the eight Cartesian coordinate vectors of a cell's surrounding nodes into eight matrix representations.
    multiplicator = jnp.array(
        [
            [
                [-1, 1, 0, 0, 0, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 1, 0, 0, 0, 0],
            ],
            [
                [-1, 1, 0, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 1, 0, 0, 0],
                [0, -1, 0, 0, 0, 1, 0, 0],
            ],
            [
                [0, 0, -1, 0, 1, 0, 0, 0],
                [-1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 1, 0],
            ],
            [
                [0, 0, 0, -1, 0, 1, 0, 0],
                [0, 0, 0, -1, 0, 0, 1, 0],
                [-1, 0, 0, 1, 0, 0, 0, 0],
            ],
            [
                [0, 0, -1, 0, 1, 0, 0, 0],
                [0, -1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1, 0, 0, 1],
            ],
            [
                [0, 0, 0, -1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 1],
                [0, -1, 0, 0, 0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, -1, 1],
                [0, 0, 0, -1, 0, 0, 1, 0],
                [0, 0, -1, 0, 0, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, -1, 1],
                [0, 0, 0, 0, 0, -1, 0, 1],
                [0, 0, 0, 0, -1, 0, 0, 1],
            ],
        ]
    )

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

    def elastic_energy_nanocell(vertices, h0, C0):
        """The elastic deformation energy of a nanocell, with respect to one of its metastable states with parameters h0 and C0.

        Parameters
        ----------
        vertices : numpy.ndarray, shape=(8, 3)
            The coordinates of the surrounding nodes (i.e. the vertices).
        h0 : numpy.ndarray, shape=(3, 3)
            The equilibrium cell matrix.
        C0 : numpy.ndarray, shape=(3, 3, 3, 3)
            The elasticity tensor.

        Returns
        -------
        energy : float
            The elastic deformation energy.

        Notes
        -----
        At first sight, the equations for bistable nanocells might seem absent from this derivation.
        They are absent here, but they have been implemented in the ``mmff.py`` script.
        This elastic deformation energy is only the energy of a single metastable state of a nanocell.
        """
        # The equations below are the same as in the default (`nanocell.py`).
        # (3.20)
        matrices = jnp.einsum("...i,ij->...j", multiplicator, vertices)

        # (3.23)
        matrices_ = jnp.einsum("...ji,kj->...ik", matrices, jnp.linalg.inv(h0))
        identity_ = jnp.array([jnp.identity(3) for _ in range(8)])
        strains = 0.5 * (
            jnp.einsum("...ji,...jk->...ik", matrices_, matrices_) - identity_
        )

        # (4.11)
        energy_density = 0.5 * jnp.einsum("...ij,ijkl,...kl", strains, C0, strains)
        energy = 0.125 * jnp.einsum("i->", energy_density) * jnp.linalg.det(h0)

        return energy

    for cell_idx in range(ncells):
        type_ = types[cell_idx]

        # Store the nodal index of each vertex of the current cell in an array.
        verts_idxs = surrounding_nodes[cell_idx]

        # Calculate the position of each vertex.
        r0 = pos[verts_idxs[0]]
        r1 = r0 + delta(verts_idxs[0], verts_idxs[1])
        r2 = r0 + delta(verts_idxs[0], verts_idxs[2])
        r3 = r0 + delta(verts_idxs[0], verts_idxs[3])
        r4 = r0 + delta(verts_idxs[0], verts_idxs[4])
        r5 = r0 + delta(verts_idxs[0], verts_idxs[5])
        r6 = r0 + delta(verts_idxs[0], verts_idxs[6])
        r7 = r0 + delta(verts_idxs[0], verts_idxs[7])
        verts = jnp.array([r0, r1, r2, r3, r4, r5, r6, r7])

        epot_states = []
        # Iterate over each metastable state.
        temp_eff = params1[int(type_)]
        for efree_state, h0_state, C0_state in zip(
            params2[int(type_)], params3[int(type_)], params4[int(type_)]
        ):
            epot_states.append(
                elastic_energy_nanocell(verts, h0_state, C0_state) + efree_state
            )
        epot_states = jnp.array(epot_states)
        epot_min = jnp.min(epot_states)
        weights_states = jnp.exp(
            -(epot_states - epot_min) / (boltzmann * temp_eff)
        )  # (3.41)
        epot_cell = epot_min - temp_eff * boltzmann * jnp.log(
            jnp.sum(weights_states)
        )  # (3.37)

        epot += epot_cell

    return epot
