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
    "CartesianDOF",
    "BaseCellDOF",
    "FullCellDOF",
    "StrainCellDOF"
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
class BaseCellDOF(DOF):
    """Fractional coordinates and cell parameters.
    
    Each subclass of BaseCellDOF considers a specific representation and subset of the cell parameters.
    The following variable names are consistently used (also in subclasses).
    
    cellvars
        An array with all variables for the cell (specific for BaseCellDOF
        subclass).
    ncellvar
        The number of cellvars (at most 9).
    celldofs
        A selection of the elements in cellvars, based on freemask.
    ncelldof
        The number of celldofs (less than or equal to ncellvar).
    frac
        Fractional coordinates.
    x
        All degrees of freedom, i.e. celldofs and frac (in that order, frac is optional).
    
    """
    def __init__(self, mmf, gpos_rms=1e-5, dpos_rms=1e-3, grvecs_rms=1e-5, drvecs_rms=1e-3, do_frozen=False, freemask=None):
        """
        Parameters
        ----------
        mmf
            A force field object.
        gpos_rms, dpos_rms, grvecs_rms, drvecs_rms
            Thresholds that define the convergence. 
            If all of the actual values drop below these thresholds, the minimizer stops.
            For each rms threshold, a corresponding max threshold is included automatically. 
            The maximum of the absolute value of a component should be smaller than 3/sqrt(N) times the rms
            threshold, where N is the number of degrees of freedom.
        do_frozen
            When True, the fractional coordinates of the atoms are kept fixed.
        freemask
            When given, this must be an array of booleans indicating which cellvars are free. 
            At least one cellvar must be free.
        
        Notes
        -----
        Convergence conditions:
        gpos_rms
            The root-mean-square of the norm of the gradients of the atoms.
        dpos_rms
            The root-mean-square of the norm of the displacements of the atoms.
        grvecs_rms
            The root-mean-square of the norm of the gradients of the cell vectors.
        drvecs_rms
            The root-mean-square of the norm of the displacements of the cell vectors.
        
        """
        if freemask is not None:
            if not (isinstance(freemask, np.ndarray) and
                    issubclass(freemask.dtype.type, np.bool_) and
                    len(freemask.shape)==1 and
                    freemask.sum() > 0):
                raise TypeError("When given, freemask must be a vector of booleans.")
        self.th_gpos_rms = gpos_rms
        self.th_dpos_rms = dpos_rms
        self.th_grvecs_rms = grvecs_rms
        self.th_drvecs_rms = drvecs_rms
        self.do_frozen = do_frozen
        self.freemask = freemask
        DOF.__init__(self, mmf)
        self._last_pos = None
        self._last_rvecs = None

    def _get_ncellvar(self):
        """The number of cellvars"""
        return len(self.domainvars0)

    ncellvar = property(_get_ncellvar)

    def _get_ncelldof(self):
        """The number of celldofs (free cellvars)"""
        if self.freemask is None:
            return len(self.domainvars0)
        else:
            return self.freemask.sum()

    ncelldof = property(_get_ncelldof)

    def _reduce_cellvars(self, cellvars):
        if self.freemask is None:
            return cellvars
        else:
            return cellvars[self.freemask]

    def _expand_celldofs(self, celldofs):
        if self.freemask is None:
            return celldofs
        else:
            cellvars = self.domainvars0.copy()
            cellvars[self.freemask] = celldofs
            return cellvars

    def _isfree(self, icellvar):
        """Returns a boolean indicating that a given cellvar is free (True) or not (False)."""
        if self.freemask is None:
            return True
        else:
            return self.freemask[icellvar]

    def _init_initial(self):
        """Set the initial value of the unknowns in x0."""
        self.domainvars0 = self._get_initial_cellvars()
        if self.freemask is not None and len(self.freemask) != self.ncellvar:
            raise TypeError("The length of the freemask vector (%i) does not "
                            "match the number of cellvars (%i)." % (
                            len(self.freemask), len(self.domainvars0)))
        celldofs0 = self._reduce_cellvars(self.domainvars0)
        gvecs_full = self.mmf.system.domain._get_gvecs(full=True)
        frac = np.dot(self.mmf.system.pos, gvecs_full.T)
        if self.do_frozen:
            self.x0 = celldofs0
            # keep the initial fractional coordinates for later use
            self._frac0 = frac
        else:
            self.x0 = np.concatenate([celldofs0, frac.ravel()])
        # Also allocate arrays for convergence testing
        self._pos = self.mmf.system.pos.copy()
        self._dpos = np.zeros(self.mmf.system.pos.shape, float)
        self._gpos = np.zeros(self.mmf.system.pos.shape, float)
        self._rvecs = self.mmf.system.domain.rvecs.copy()
        self._dcell = np.zeros(self._rvecs.shape, float)
        self._vtens = np.zeros((3, 3), float)
        self._grvecs = np.zeros(self._rvecs.shape, float)

    def _update(self, x):
        self._rvecs = self._cellvars_to_rvecs(self._expand_celldofs(x[:self.ncelldof]))
        self.mmf.update_rvecs(self._rvecs[:])
        rvecs_full = self.mmf.system.domain._get_rvecs(full=True)
        if self.do_frozen:
            frac = self._frac0
        else:
            frac = x[self.ncelldof:].reshape(-1,3)
        self._pos[:] = np.dot(frac, rvecs_full)
        self.mmf.update_pos(self._pos[:])

    def fun(self, x, do_gradient=False):
        """Computes the energy and optionally the gradient.
        
        Parameters
        ----------
        x
            All degrees of freedom.
        do_gradient
            When True, the gradient is also returned.
        
        """
        self._update(x)
        if do_gradient:
            self._gpos[:] = 0.0
            self._vtens[:] = 0.0
            v = self.mmf.compute(self._gpos, self._vtens)
            # the derivatives of the energy toward the cell vector components
            self._grvecs[:] = np.dot(self.mmf.system.domain.gvecs, self._vtens)
            # the derivative of the energy toward the celldofs
            jacobian = self._get_celldofs_jacobian(x[:self.ncelldof])
            assert jacobian.shape[0] == self._grvecs.size
            assert jacobian.shape[1] == self.ncelldof
            self._gx[:self.ncelldof] = np.dot(self._grvecs.ravel(), jacobian)
            # project out components from grvecs that are not affected by gcelldofs
            U, S, Vt = np.linalg.svd(jacobian, full_matrices=False)
            self._grvecs[:] =  np.dot(U, np.dot(U.T, self._grvecs.ravel())).reshape(-1, 3)
            if not self.do_frozen:
                self._gx[self.ncelldof:] = np.dot(self._gpos, self._rvecs.T).ravel()
            return v, self._gx.copy()
        else:
            return self.mmf.compute()

    def check_convergence(self):
        # When called for the first time, initialize _last_pos and _last_rvecs
        if self._last_pos is None:
            self._last_pos = self._pos.copy()
            self._last_rvecs = self._rvecs.copy()
            self.converged = False
            self.conv_val = 2
            self.conv_worst = "first_step"
            self.conv_count = -1
            return
        # Compute the values that have to be compared to the thresholds
        if not self.do_frozen:
            gpossq = (self._gpos**2).sum(axis=1)
            self.gpos_max = np.sqrt(gpossq.max())
            self.gpos_rms = np.sqrt(gpossq.mean())
            self._dpos[:] = self._pos
            self._dpos -= self._last_pos
            self.gpos_indmax = gpossq.argmax()
        #
        dpossq = (self._dpos**2).sum(axis=1)
        self.dpos_max = np.sqrt(dpossq.max())
        self.dpos_rms = np.sqrt(dpossq.mean())
        #
        grvecssq = (self._grvecs**2).sum(axis=1)
        self.grvecs_max = np.sqrt(grvecssq.max())
        self.grvecs_rms = np.sqrt(grvecssq.mean())
        self._dcell[:] = self._rvecs
        self._dcell -= self._last_rvecs
        #
        dcellsq = (self._dcell**2).sum(axis=1)
        self.drvecs_max = np.sqrt(dcellsq.max())
        self.drvecs_rms = np.sqrt(dcellsq.mean())
        # Compute a general value that has to go below 1.0 to have convergence.
        conv_vals = []
        if not self.do_frozen and self.th_gpos_rms is not None:
            conv_vals.append((self.gpos_rms/self.th_gpos_rms, "gpos_rms"))
            conv_vals.append((self.gpos_max/(self.th_gpos_rms*3), "gpos_max(%i)" %self.gpos_indmax))
        if self.th_dpos_rms is not None:
            conv_vals.append((self.dpos_rms/self.th_dpos_rms, "dpos_rms"))
            conv_vals.append((self.dpos_max/(self.th_dpos_rms*3), "dpos_max"))
        if self.th_grvecs_rms is not None:
            conv_vals.append((self.grvecs_rms/self.th_grvecs_rms, "grvecs_rms"))
            conv_vals.append((self.grvecs_max/(self.th_grvecs_rms*3), "grvecs_max"))
        if self.th_drvecs_rms is not None:
            conv_vals.append((self.drvecs_rms/self.th_drvecs_rms, "drvecs_rms"))
            conv_vals.append((self.drvecs_max/(self.th_drvecs_rms*3), "drvecs_max"))
        if len(conv_vals) == 0:
            raise RuntimeError("At least one convergence criterion must be present.")
        self.conv_val, self.conv_worst = max(conv_vals)
        self.conv_count = sum(int(v>=1) for v, n in conv_vals)
        self.converged = (self.conv_count == 0)
        self._last_pos[:] = self._pos[:]
        self._last_rvecs[:] = self._rvecs[:]

    def log(self):
        rvecs = self.mmf.system.domain.rvecs
        lengths, angles = self.mmf.system.domain.parameters
        rvec_names = "abc"
        angle_names = ["alpha", "beta", "gamma"]
        log(" ")
        log("Final Unit Cell:")
        log("----------------")
        log("- cell vectors:")
        for i in range(len(rvecs)):
            log("    %s = %s %s %s" %(rvec_names[i], log.length(rvecs[i,0]), log.length(rvecs[i,1]), log.length(rvecs[i,2]) ))
        log(" ")
        log("- lengths, angles and volume:")
        for i in range(len(rvecs)):
            log("    |%s|  = %s" % (rvec_names[i], log.length(lengths[i])))
        for i in range(len(angles)):
            log("    %5s = %s" % (angle_names[i], log.angle(angles[i])))
        log("    volume = %s" % log.volume(self.mmf.system.domain.volume) )

    def _get_initial_cellvars(self):
        """Return the initial values of all cellvars."""
        raise NotImplementedError

    def _cellvars_to_rvecs(self, cellvars):
        """Convert cellvars to cell rvecs."""
        raise NotImplementedError

    def _get_celldofs_jacobian(self, x):
        """Return the jacobian of the function rvecs(celldofs).
           
        Rows correspond to cell vector components. 
        Columns correspond to celldofs. 
        There should never be more columns than rows.
        
        """
        raise NotImplementedError


class FullCellDOF(BaseCellDOF):
    """DOF that includes all 9 components of the cell vectors.
    
    The degrees of freedom are rescaled cell vectors ordered in one row:
    3D periodic: [a_x/s, a_y/s, a_z/s, b_x/s, b_y/s, b_z/s, c_x/s, c_y/s, c_z/s] 
        where s is the cube root of the initial cell volume such that the cell DOFs become dimensionless.
    2D periodic: [a_x/s, a_y/s, a_z/s, b_x/s, b_y/s, b_z/s] 
        where s is the square root of the initial cell surface such that the cell DOFs become dimensionless.
    1D periodic: [a_x/s, a_y/s, a_z/s] 
        where s is the length of the initial cell vector such that the cell DOFs become dimensionless.
    
    """
    def _get_initial_cellvars(self):
        cell = self.mmf.system.domain
        if cell.nvec == 0:
            raise ValueError("A cell optimization requires a system that is periodic.")
        self._rvecs_scale = cell.volume**(1.0/cell.nvec)
        return cell.rvecs.ravel()/self._rvecs_scale

    def _cellvars_to_rvecs(self, cellvars):
        return cellvars.reshape(-1, 3)*self._rvecs_scale

    def _get_celldofs_jacobian(self, x):
        result = np.identity(self.ncellvar)*self._rvecs_scale
        if self.freemask is not None:
            result = result[:,self.freemask]
        return result
        

class StrainCellDOF(BaseCellDOF):
    """Eliminates rotations of the unit cell. thus six cell parameters are free.
    
    The degrees of freedom are coefficients in symmetrix matrix
    transformation, A, that is applied to  the initial cell vectors.
    
    3D periodic: [A_00, A_11, A_22, 2*A_12, 2*A_20, 2*A_01]
    2D periodic: [A_00, A_11, 2*A_01]
    1D periodic: [A_00]
    
    """
    def _get_initial_cellvars(self):
        cell = self.mmf.system.domain
        if cell.nvec == 0:
            raise ValueError("A cell optimization requires a system that is periodic.")
        self.rvecs0 = cell.rvecs.copy()
        if cell.nvec == 3:
            return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        elif cell.nvec == 2:
            return np.array([1.0, 1.0, 0.0])
        elif cell.nvec == 1:
            return np.array([1.0])
        else:
            raise NotImplementedError

    def _cellvars_to_rvecs(self, x):
        nvec = self.mmf.system.domain.nvec
        scales = x[:(nvec*(nvec+1))//2]
        if nvec == 3:
            deform = np.array([
                [    scales[0], 0.5*scales[5], 0.5*scales[4]],
                [0.5*scales[5],     scales[1], 0.5*scales[3]],
                [0.5*scales[4], 0.5*scales[3],     scales[2]],
            ])
        elif nvec == 2:
            deform = np.array([
                [    scales[0], 0.5*scales[2]],
                [0.5*scales[2],     scales[1]],
            ])
        elif nvec == 1:
            deform = np.array([[scales[0]]])
        else:
            raise NotImplementedError
        return np.dot(deform, self.rvecs0)

    def _get_celldofs_jacobian(self, x):
        cols = []
        nvec = self.mmf.system.domain.nvec
        if nvec == 3:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2],
                             0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0])
            if self._isfree(1):
                cols.append([0.0, 0.0, 0.0,
                             self.rvecs0[1,0], self.rvecs0[1,1], self.rvecs0[1,2],
                             0.0, 0.0, 0.0])
            if self._isfree(2):
                cols.append([0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0,
                             self.rvecs0[2,0], self.rvecs0[2,1], self.rvecs0[2,2]])
            if self._isfree(3):
                cols.append([0.0, 0.0, 0.0,
                             self.rvecs0[2,0]/2, self.rvecs0[2,1]/2, self.rvecs0[2,2]/2,
                             self.rvecs0[1,0]/2, self.rvecs0[1,1]/2, self.rvecs0[1,2]/2])
            if self._isfree(4):
                cols.append([self.rvecs0[2,0]/2, self.rvecs0[2,1]/2, self.rvecs0[2,2]/2,
                             0.0, 0.0, 0.0,
                             self.rvecs0[0,0]/2, self.rvecs0[0,1]/2, self.rvecs0[0,2]/2])
            if self._isfree(5):
                cols.append([self.rvecs0[1,0]/2, self.rvecs0[1,1]/2, self.rvecs0[1,2]/2,
                             self.rvecs0[0,0]/2, self.rvecs0[0,1]/2, self.rvecs0[0,2]/2,
                             0.0, 0.0, 0.0])
        elif nvec == 2:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2],
                             0.0, 0.0, 0.0])
            if self._isfree(1):
                cols.append([0.0, 0.0, 0.0,
                             self.rvecs0[1,0], self.rvecs0[1,1], self.rvecs0[1,2]])
            if self._isfree(2):
                cols.append([self.rvecs0[1,0]/2, self.rvecs0[1,1]/2, self.rvecs0[1,2]/2,
                             self.rvecs0[0,0]/2, self.rvecs0[0,1]/2, self.rvecs0[0,2]/2])
        else:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2]])
        return np.array(cols).T


