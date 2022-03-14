#!/usr/bin/env python

"""
Abstraction layer for degrees of freedom.
All these classes are called DOF classes, because they specify a set of
degrees of freedom. These DOF classes are used for geometry/domain optimization
and harmonic approximations.
"""

import numpy as np

from molmod.minimizer import check_delta

from micmec.log import log


__all__ = [
    "DOF", "CartesianDOF", "BaseDomainDOF", "FullDomainDOF", "StrainDomainDOF",
    "IsoDomainDOF", "AnisoDomainDOF", "FixedBCDOF", "FixedVolOrthoDomainDOF",
]


class DOF(object):
    def __init__(self, mmf):
        """
        **ARGUMENTS**
        mmf
            A MicroMechanicalField instance.
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
    """Cartesian degrees of freedom
       This DOF is also applicable to periodic systems. Domain parameters are not
       modified when this DOF is used.
    """
    def __init__(self, mmf, gpos_rms=1e-5, dpos_rms=1e-3, select=None):
        """
        **ARGUMENTS**
        mmf
            A MicroMechanicalField instance.

        **OPTIONAL ARGUMENTS**
        gpos_rms, dpos_rms
            Thresholds that define the convergence. If all of the actual
            values drop below these thresholds, the minimizer stops.
            For each rms threshold, a corresponding max threshold is
            included automatically. The maximum of the absolute value of a
            component should be smaller than 3/sqrt(N) times the rms
            threshold, where N is the number of degrees of freedom.
        select
            A selection of atoms whose degrees of freedom are included. If
            not list is given, all atomic coordinates are included.

        **CONVERGENCE CONDITIONS**
        gpos_rms
            The root-mean-square of the norm of the gradients of the atoms.
        dpos_rms
            The root-mean-square of the norm of the displacements of the
            atoms.
        """
        self.th_gpos_rms = gpos_rms
        self.th_dpos_rms = dpos_rms
        self.select = select
        DOF.__init__(self, mmf)
        self._last_pos = None

    def _init_initial(self):
        """Set the initial value of the unknowns in x0"""
        if self.select is None:
            self.x0 = mmf.system.pos.ravel().copy()
        else:
            self.x0 = mmf.system.pos[self.select].ravel().copy()
        # Keep a copy of the current positions for later use
        self._pos = mmf.system.pos.copy()
        # Allocate arrays for atomic displacements and gradients
        self._dpos = np.zeros(mmf.system.pos.shape, float)
        self._gpos = np.zeros(mmf.system.pos.shape, float)

    def _update(self, x):
        if self.select is None:
            self._pos[:] = x.reshape(-1,3)
        else:
            self._pos[self.select] = x.reshape(-1,3)
        mmf.update_pos(self._pos[:])

    def fun(self, x, do_gradient=False):
        """Computes the energy and optionally the gradient.
           **ARGUMENTS**
           x
                The degrees of freedom
           **OPTIONAL ARGUMENTS**
           do_gradient
                When True, the gradient is also returned.
        """
        self._update(x)
        if do_gradient:
            self._gpos[:] = 0.0
            v = mmf.compute(self._gpos)
            if self.select is None:
                self._gx[:] = self._gpos.ravel()
            else:
                self._gx[:] = self._gpos[self.select].ravel()
            return v, self._gx.copy()
        else:
            return mmf.compute()

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


class BaseDomainDOF(DOF):
    """Fractional coordinates and domain parameters
       Several subclasses of BaseDomainDOF are implemented below. Each one
       considers a specific representation and subset of the domain parameters.
       The following variable names are consistently used (also in subclasses):
       domainvars
            An array with all variables for the domain (specific for ja BaseDomainDOF
            subclass).
       ndomainvar
            The number of domainvars (at most 9).
       domaindofs
            A selection of the elements in domainvars, based on freemask.
       ndomaindof
            The number of domaindofs (less than or equal to ndomainvar).
       frac
            Fractional coordinates.
       x
            All degrees of freedom, i.e. domaindofs and frac (in that order, frac
            is optional).
       The suffix 0
            Used for initial values of something.
    """
    def __init__(self, mmf, gpos_rms=1e-5, dpos_rms=1e-3, grvecs_rms=1e-5, drvecs_rms=1e-3, do_frozen=False, freemask=None):
        """
        **ARGUMENTS**
        mmf
            A MicroMechanicalField instance.
        
        **OPTIONAL ARGUMENTS**
        gpos_rms, dpos_rms, grvecs_rms, drvecs_rms
            Thresholds that define the convergence. If all of the actual
            values drop below these thresholds, the minimizer stops.
            For each rms threshold, a corresponding max threshold is
            included automatically. The maximum of the absolute value of a
            component should be smaller than 3/sqrt(N) times the rms
            threshold, where N is the number of degrees of freedom.
        do_frozen
            When True, the fractional coordinates of the atoms are kept
            fixed.
        freemask
            When given, this must be an array of booleans indicating which
            domainvars are free. At least one domainvar must be free.
        **Convergence conditions:**
        gpos_rms
            The root-mean-square of the norm of the gradients of the atoms.
        dpos_rms
            The root-mean-square of the norm of the displacements of the
            atoms.
        grvecs_rms
            The root-mean-square of the norm of the gradients of the domain
            vectors.
        drvecs_rms
            The root-mean-square of the norm of the displacements of the
            domain vectors.
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

    def _get_ndomainvar(self):
        """The number of domainvars."""
        return len(self.domainvars0)

    ndomainvar = property(_get_ndomainvar)

    def _get_ndomaindof(self):
        """The number of domaindofs (free domainvars)."""
        if self.freemask is None:
            return len(self.domainvars0)
        else:
            return self.freemask.sum()

    ndomaindof = property(_get_ndomaindof)

    def _reduce_domainvars(self, domainvars):
        if self.freemask is None:
            return domainvars
        else:
            return domainvars[self.freemask]

    def _expand_domaindofs(self, domaindofs):
        if self.freemask is None:
            return domaindofs
        else:
            domainvars = self.domainvars0.copy()
            domainvars[self.freemask] = domaindofs
            return domainvars

    def _isfree(self, idomainvar):
        """Returns a boolean indicating that a given domainvar is free (True) or not (False)."""
        if self.freemask is None:
            return True
        else:
            return self.freemask[idomainvar]

    def _init_initial(self):
        """Set the initial value of the unknowns in x0."""
        self.domainvars0 = self._get_initial_domainvars()
        if self.freemask is not None and len(self.freemask) != self.ndomainvar:
            raise TypeError("The length of the freemask vector (%i) does not "
                            "match the number of domainvars (%i)." % (
                            len(self.freemask), len(self.domainvars0)))
        domaindofs0 = self._reduce_domainvars(self.domainvars0)
        gvecs_full = mmf.system.domain._get_gvecs(full=True)
        frac = np.dot(mmf.system.pos, gvecs_full.T)
        if self.do_frozen:
            self.x0 = domaindofs0
            # keep the initial fractional coordinates for later use
            self._frac0 = frac
        else:
            self.x0 = np.concatenate([domaindofs0, frac.ravel()])
        # Also allocate arrays for convergence testing
        self._pos = mmf.system.pos.copy()
        self._dpos = np.zeros(mmf.system.pos.shape, float)
        self._gpos = np.zeros(mmf.system.pos.shape, float)
        self._rvecs = mmf.system.domain.rvecs.copy()
        self._ddomain = np.zeros(self._rvecs.shape, float)
        self._vtens = np.zeros((3, 3), float)
        self._grvecs = np.zeros(self._rvecs.shape, float)

    def _update(self, x):
        self._rvecs = self._domainvars_to_rvecs(self._expand_domaindofs(x[:self.ndomaindof]))
        mmf.update_rvecs(self._rvecs[:])
        rvecs_full = mmf.system.domain._get_rvecs(full=True)
        if self.do_frozen:
            frac = self._frac0
        else:
            frac = x[self.ndomaindof:].reshape(-1,3)
        self._pos[:] = np.dot(frac, rvecs_full)
        mmf.update_pos(self._pos[:])

    def fun(self, x, do_gradient=False):
        """
        Computes the energy and optionally the gradient.

        **ARGUMENTS**
        x
            All degrees of freedom.

        **OPTIONAL ARGUMENTS**
        do_gradient
            When True, the gradient is also returned.
        """
        self._update(x)
        if do_gradient:
            self._gpos[:] = 0.0
            self._vtens[:] = 0.0
            v = mmf.compute(self._gpos, self._vtens)
            # the derivatives of the energy toward the domain vector components
            self._grvecs[:] = np.dot(mmf.system.domain.gvecs, self._vtens)
            # the derivative of the energy toward the domaindofs
            jacobian = self._get_domaindofs_jacobian(x[:self.ndomaindof])
            assert jacobian.shape[0] == self._grvecs.size
            assert jacobian.shape[1] == self.ndomaindof
            self._gx[:self.ndomaindof] = np.dot(self._grvecs.ravel(), jacobian)
            # project out components from grvecs that are not affected by gdomaindofs
            U, S, Vt = np.linalg.svd(jacobian, full_matrices=False)
            self._grvecs[:] =  np.dot(U, np.dot(U.T, self._grvecs.ravel())).reshape(-1, 3)
            if not self.do_frozen:
                self._gx[self.ndomaindof:] = np.dot(self._gpos, self._rvecs.T).ravel()
            return v, self._gx.copy()
        else:
            return mmf.compute()

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
        self._ddomain[:] = self._rvecs
        self._ddomain -= self._last_rvecs
        #
        ddomainsq = (self._ddomain**2).sum(axis=1)
        self.drvecs_max = np.sqrt(ddomainsq.max())
        self.drvecs_rms = np.sqrt(ddomainsq.mean())
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
        rvecs = mmf.system.domain.rvecs
        lengths, angles = mmf.system.domain.parameters
        rvec_names = "abc"
        angle_names = ["alpha", "beta", "gamma"]
        log(" ")
        log("Final Unit Domain:")
        log("----------------")
        log("- domain vectors:")
        for i in range(len(rvecs)):
            log("    %s = %s %s %s" %(rvec_names[i], log.length(rvecs[i,0]), log.length(rvecs[i,1]), log.length(rvecs[i,2]) ))
        log(" ")
        log("- lengths, angles and volume:")
        for i in range(len(rvecs)):
            log("    |%s|  = %s" % (rvec_names[i], log.length(lengths[i])))
        for i in range(len(angles)):
            log("    %5s = %s" % (angle_names[i], log.angle(angles[i])))
        log("    volume = %s" % log.volume(mmf.system.domain.volume) )

    def _get_initial_domainvars(self):
        """Return the initial values of all domainvars."""
        raise NotImplementedError

    def _domainvars_to_rvecs(self, domainvars):
        """Convert domainvars to domain rvecs."""
        raise NotImplementedError

    def _get_domaindofs_jacobian(self, x):
        """
        Return the jacobian of the function rvecs(domaindofs).
        Rows correspond to domain vector components. Collumns correspond to
        domaindofs. There should never be more columns than rows.
        """
        raise NotImplementedError


class FullDomainDOF(BaseDomainDOF):
    """
    DOF that includes all 9 components of the domain vectors
    The degrees of freedom are rescaled domain vectors ordered in one row:
    * 3D periodic: [a_x/s, a_y/s, a_z/s, b_x/s, b_y/s, b_z/s, c_x/s, c_y/s,
     c_z/s] where s is the cube root of the initial domain volume such that
     the domain DOFs become dimensionless.
    * 2D periodic: [a_x/s, a_y/s, a_z/s, b_x/s, b_y/s, b_z/s] where s is the
     square root of the initial domain surface such that the domain DOFs become
     dimensionless.
    * 1D periodic: [a_x/s, a_y/s, a_z/s] where s is the length of the initial
     domain vector such that the domain DOFs become dimensionless.
    """
    def _get_initial_domainvars(self):
        domain = mmf.system.domain
        if domain.nvec == 0:
            raise ValueError("A domain optimization requires a system that is periodic.")
        self._rvecs_scale = domain.volume**(1.0/domain.nvec)
        return domain.rvecs.ravel()/self._rvecs_scale

    def _domainvars_to_rvecs(self, domainvars):
        return domainvars.reshape(-1, 3)*self._rvecs_scale

    def _get_domaindofs_jacobian(self, x):
        result = np.identity(self.ndomainvar)*self._rvecs_scale
        if self.freemask is not None:
            result = result[:,self.freemask]
        return result


class StrainDomainDOF(BaseDomainDOF):
    """
    Eliminates rotations of the unit domain. thus six domain parameters are free.
    The degrees of freedom are coefficients in symmetrix matrix
    transformation, A, that is applied to  the initial domain vectors.
    * 3D periodic: [A_00, A_11, A_22, 2*A_12, 2*A_20, 2*A_01]
    * 2D periodic: [A_00, A_11, 2*A_01]
    * 1D periodic: [A_00]
    Why does this work? Let R be the array with domain vectors as rows. It can
    always be written as a product,
        R = R_0.F,
    where F is an arbitrary 3x3 matrix. Application of SVD to the matrix F
    yields:
        R = R_0.U.S.V^T = R_0.U.V^T.V.S.V^T
    Then W=U.V^T is a orthonormal matrix and A=V.S.V^T is a symmetric matrix.
    The orthonormal matrix W is merely a rotation of the domain vectors, which
    can be omitted as the internal energy is invariant under such rotations.
    The symmetric matrix actually deforms the domain and is the part of interest.
    """
    def _get_initial_domainvars(self):
        domain = mmf.system.domain
        if domain.nvec == 0:
            raise ValueError("A domain optimization requires a system that is periodic.")
        self.rvecs0 = domain.rvecs.copy()
        if domain.nvec == 3:
            return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        elif domain.nvec == 2:
            return np.array([1.0, 1.0, 0.0])
        elif domain.nvec == 1:
            return np.array([1.0])
        else:
            raise NotImplementedError

    def _domainvars_to_rvecs(self, x):
        nvec = mmf.system.domain.nvec
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

    def _get_domaindofs_jacobian(self, x):
        cols = []
        nvec = mmf.system.domain.nvec
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


class AnisoDomainDOF(BaseDomainDOF):
    """
    Only the lengths of the domain vectors are free. angles are fixed.
    The degrees of freedom are dimensionless scale factors for the domain
    lengths, using the initial domain vectors as the reference point. (This is
    one DOF per periodic dimension.)
    """
    def _get_initial_domainvars(self):
        domain = mmf.system.domain
        if domain.nvec == 0:
            raise ValueError("A domain optimization requires a system that is periodic.")
        self.rvecs0 = domain.rvecs.copy()
        return np.ones(domain.nvec, float)

    def _domainvars_to_rvecs(self, x):
        nvec = mmf.system.domain.nvec
        return self.rvecs0*x[:nvec, None]

    def _get_domaindofs_jacobian(self, x):
        cols = []
        nvec = mmf.system.domain.nvec
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
        elif nvec == 2:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2],
                             0.0, 0.0, 0.0])
            if self._isfree(1):
                cols.append([0.0, 0.0, 0.0,
                             self.rvecs0[1,0], self.rvecs0[1,1], self.rvecs0[1,2]])
        else:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2]])
        return np.array(cols).T



class IsoDomainDOF(BaseDomainDOF):
    """
    The domain is only allowed to undergo isotropic scaling
    The only degree of freedom is an isotropic scaling factor, using the
    initial domain vectors as a reference.
    """
    def _get_initial_domainvars(self):
        domain = mmf.system.domain
        if domain.nvec == 0:
            raise ValueError("A domain optimization requires a system that is periodic.")
        self.rvecs0 = domain.rvecs.copy()
        return np.ones(1, float)

    def _domainvars_to_rvecs(self, x):
        return self.rvecs0*x[0]

    def _get_domaindofs_jacobian(self, x):
        cols = []
        nvec = mmf.system.domain.nvec
        if nvec == 3:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2],
                             self.rvecs0[1,0], self.rvecs0[1,1], self.rvecs0[1,2],
                             self.rvecs0[2,0], self.rvecs0[2,1], self.rvecs0[2,2]])
        elif nvec == 2:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2],
                             self.rvecs0[1,0], self.rvecs0[1,1], self.rvecs0[1,2]])
        else:
            if self._isfree(0):
                cols.append([self.rvecs0[0,0], self.rvecs0[0,1], self.rvecs0[0,2]])
        return np.array(cols).T


class FixedBCDOF(BaseDomainDOF):
    """
    A rectangular domain that can only stretch along one axis
    This domain optimization constrains the domain in the y and z direction to the
    original values, but allows expansion and contraction in the x direction.
    The system should be rotated such that the initial domain vectors look like::
        a = ( ax , 0  , 0  )
        b = ( 0  , by , bz )
        c = ( 0  , cy , cz )
    During optimization, only ax will be allowed to change.
    This type of constraint can be used when looking at a structure that is
    periodic only in one dimension, but you have to fake a 3D structure to
    be able to use Ewald summation
    """
    def _get_initial_domainvars(self):
        domain = mmf.system.domain
        if domain.nvec != 3:
            raise ValueError("FixedBCDOF requires a 3D periodic domain.")
        self.rvecs0 = domain.rvecs.copy()
        if not (self.rvecs0[1, 0] == 0.0 and self.rvecs0[0, 1] == 0.0 and
                self.rvecs0[2, 0] == 0.0 and self.rvecs0[0, 2] == 0.0):
            raise ValueError("FixedBCDOF requires the follow domain vector components to be zero: ay, az, bx and cx.")
        return np.ones(1, float)

    def _domainvars_to_rvecs(self, x):
        # Copy original rvecs
        rvecs = self.rvecs0.copy()
        # Update value for ax
        rvecs[0,0] = x[0]*rvecs[0,0]
        return rvecs

    def _get_domaindofs_jacobian(self, x):
        return np.array([[self.rvecs0[0,0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T


class FixedVolOrthoDomainDOF(BaseDomainDOF):
    """
    Orthorombic domain optimizer with a fixed volume.
    These constraints are implemented by using the following domain vectors::
       a = (  s*a0*la  ,     0     ,      0         )
       b = (     0     ,  s*b0*lb  ,      0         )
       c = (     0     ,     0     ,  s*c0/(la*lb)  )
    with s = (V/V0)^(1/3)
    """
    def __init__(self, mmf, volume=None, gpos_rms=1e-5, dpos_rms=1e-3, grvecs_rms=1e-5, drvecs_rms=1e-3, do_frozen=False, freemask=None):
        """
        **OPTIONAL ARGUMENTS (in addition to those of BaseDomainDOF):**
        volume
            The desired volume of the domain. (When not given, the current
            volume of the system is not altered.)
        """
        self.volume = volume
        BaseDomainDOF.__init__(self, mmf, gpos_rms, dpos_rms, grvecs_rms, drvecs_rms, do_frozen, freemask)

    def _get_initial_domainvars(self):
        domain = mmf.system.domain
        if domain.nvec != 3:
            raise ValueError("FixedVolOrthDomainDOF requires a 3D periodic domain")
        self.rvecs0 = domain.rvecs.copy()
        if not (self.rvecs0[1, 0] == 0.0 and self.rvecs0[0, 1] == 0.0 and
                self.rvecs0[2, 0] == 0.0 and self.rvecs0[0, 2] == 0.0 and
                self.rvecs0[1, 2] == 0.0 and self.rvecs0[2, 1] == 0.0):
            raise ValueError("FixedVolOrthDomainDOF requires the follow domain vector components to be zero: ay, az, bx, bz, cx and cy.")
        if self.volume is not None:
            self.rvecs0 *= (self.volume/domain.volume)**(1.0/3.0)
        return np.array([1.0, 1.0])

    def _domainvars_to_rvecs(self, x):
        rvecs = np.zeros([3,3], float)
        rvecs[0,0] = self.rvecs0[0,0]*x[0]
        rvecs[1,1] = self.rvecs0[1,1]*x[1]
        rvecs[2,2] = self.rvecs0[2,2]/(x[0]*x[1])
        return rvecs

    def _get_domaindofs_jacobian(self, x):
        cols = []
        if self._isfree(0):
            cols.append([self.rvecs0[0,0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -self.rvecs0[0,0]/x[1]/x[0]**2])
        if self._isfree(1):
            cols.append([0.0, 0.0, 0.0, 0.0, self.rvecs0[1,1], 0.0, 0.0, 0.0, -self.rvecs0[0,0]/x[0]/x[1]**2])
        return np.array(cols).T


