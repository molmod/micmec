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


"""Abstraction layer for degrees of freedom.

All these classes are called DOF classes, because they specify a set of degrees of freedom. 
These DOF classes are used for geometry/domain optimization and harmonic approximations.
"""

import numpy as np

from molmod.minimizer import check_delta

from micmec.log import log

__all__ = [
    "DOF", 
    "CartesianDOF"
]


class DOF(object):
    def __init__(self, mmf):
        """
        Parameters
        ----------
        mmf : micmec.pes.mmff.MicMecForceField object
            The micromechanical force field.
        
        """
        self.mmf = mmf
        self.x0 = None
        self._init_initial()
        self._gx = np.zeros(self.ndof, float)

    def _init_initial(self):
        """Set the initial value of the unknowns in x0."""
        raise NotImplementedError

    ndof = property(lambda self: len(self.x0))

    def _update(self, x):
        raise NotImplementedError

    def reset(self):
        self._update(self.x0)

    def check_delta(self, x=None, eps=1e-4, zero=None):
        """Test the analytical derivatives."""
        if x is None:
            x = self.x0
        dxs = np.random.uniform(-eps, eps, (100, len(x)))
        if zero is not None:
            dxs[:, zero] = 0.0
        check_delta(self.fun, x, dxs)

    def log(self):
        pass


class CartesianDOF(DOF):
    """Cartesian degrees of freedom.

    This DOF is also applicable to periodic systems. 
    Domain parameters are not modified when this DOF is used.
    """
    def __init__(self, mmf, gpos_rms=1e-5, dpos_rms=1e-3, select=None):
        """
        Parameters
        ----------
        mmf : micmec.pes.mmff.MicMecForceField object
            The micromechanical force field.
        gpos_rms, dpos_rms : float, optional
            Thresholds that define the convergence. 
            If all of the actual values drop below these thresholds, the minimizer stops.
            For each rms threshold, a corresponding max threshold is included automatically. 
            The maximum of the absolute value of a component should be smaller than 3/sqrt(N) times the rms threshold, 
            where N is the number of degrees of freedom.
        select : list
            A selection of nodes whose degrees of freedom are included. 
            If no list is given, all nodal coordinates are included.

        Notes
        -----
        Convergence conditions:
        gpos_rms
            The root-mean-square of the norm of the gradients of the nodes.
        dpos_rms
            The root-mean-square of the norm of the displacements of the nodes.

        """
        self.th_gpos_rms = gpos_rms
        self.th_dpos_rms = dpos_rms
        self.select = select
        DOF.__init__(self, mmf)
        self._last_pos = None

    def _init_initial(self):
        """Set the initial value of the unknowns in x0"""
        if self.select is None:
            self.x0 = self.mmf.system.pos.ravel().copy()
        else:
            self.x0 = self.mmf.system.pos[self.select].ravel().copy()
        # Keep a copy of the current positions for later use
        self._pos = self.mmf.system.pos.copy()
        # Allocate arrays for nodal displacements and gradients
        self._dpos = np.zeros(self.mmf.system.pos.shape, float)
        self._gpos = np.zeros(self.mmf.system.pos.shape, float)

    def _update(self, x):
        if self.select is None:
            self._pos[:] = x.reshape(-1,3)
        else:
            self._pos[self.select] = x.reshape(-1,3)
        self.mmf.update_pos(self._pos[:])

    def fun(self, x, do_gradient=False):
        """Computes the energy and optionally the gradient.

        Parameters
        ----------
        x : numpy.ndarray
            All degrees of freedom.
        do_gradient : bool, optional
            When True, the gradient is also returned.
        
        """
        self._update(x)
        if do_gradient:
            self._gpos[:] = 0.0
            v = self.mmf.compute(self._gpos)
            if self.select is None:
                self._gx[:] = self._gpos.ravel()
            else:
                self._gx[:] = self._gpos[self.select].ravel()
            return v, self._gx.copy()
        else:
            return self.mmf.compute()

    def check_convergence(self):
        # When called for the first time, initialize _last_pos
        if self._last_pos is None:
            self._last_pos = self._pos.copy()
            self.converged = False
            self.conv_val = 2
            self.conv_worst = "first_step"
            self.conv_count = -1
            return
        # Compute the values that have to be compared to the thresholds
        if self.select is None:
            gpossq = (self._gpos**2).sum(axis=1)
        else:
            gpossq = (self._gpos[self.select]**2).sum(axis=1)
        self.gpos_max = np.sqrt(gpossq.max())
        self.gpos_rms = np.sqrt(gpossq.mean())
        #
        self._dpos[:] = self._pos
        self._dpos -= self._last_pos
        if self.select is None:
            dpossq = (self._dpos**2).sum(axis=1)
        else:
            dpossq = (self._dpos[self.select]**2).sum(axis=1)
        self.dpos_max = np.sqrt(dpossq.max())
        self.dpos_rms = np.sqrt(dpossq.mean())
        # Compute a general value that has to go below 1.0 to have convergence.
        conv_vals = []
        if self.th_gpos_rms is not None:
            conv_vals.append((self.gpos_rms/self.th_gpos_rms, "gpos_rms"))
            conv_vals.append((self.gpos_max/(self.th_gpos_rms*3), "gpos_max"))
        if self.th_dpos_rms is not None:
            conv_vals.append((self.dpos_rms/self.th_dpos_rms, "dpos_rms"))
            conv_vals.append((self.dpos_max/(self.th_dpos_rms*3), "dpos_max"))
        if len(conv_vals) == 0:
            raise RuntimeError("At least one convergence criterion must be present.")
        self.conv_val, self.conv_worst = max(conv_vals)
        self.conv_count = sum(int(v>=1) for v, n in conv_vals)
        self.converged = (self.conv_count == 0)
        self._last_pos[:] = self._pos[:]


# Yaff includes even more DOF objects.
# Feel free to add those objects, their implementation should only require minor adjustments.

