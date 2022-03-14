#!/usr/bin/env python

import numpy as np
cimport numpy as np
cimport cell

from micmec.log import log


__all__ = ["Domain"]

cdef class Domain:
    """
    Representation of periodic boundary conditions.
    0, 1, 2 and 3 dimensional systems are supported. 
    The domain vectors need not to be orthogonal.
    """
    cdef cell.cell_type* _c_domain

    def __cinit__(self, *args, **kwargs):
        self._c_domain = cell.cell_new()
        if self._c_domain is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_domain is not NULL:
            cell.cell_free(self._c_domain)

    def __init__(self, np.ndarray[double, ndim=2] rvecs):
        """
        **ARGUMENTS**
        rvecs
            A numpy array with at most three domain vectors, layed out as
            rows in a rank-2 matrix. For non-periodic systems, this array
            must have shape (0,3).
        """
        self.update_rvecs(rvecs)

    def update_rvecs(self, np.ndarray[double, ndim=2] rvecs):
        """
        Change the domain vectors and recompute the reciprocal domain vectors.

        **ARGUMENTS**
        rvecs
            A numpy array with at most three domain vectors, layed out as
            rows in a rank-2 matrix. For non-periodic systems, this array
            must have shape (0,3).
        """
        cdef np.ndarray[double, ndim=2] mod_rvecs
        cdef np.ndarray[double, ndim=2] gvecs
        cdef int nvec
        if rvecs is None or rvecs.size == 0:
            mod_rvecs = np.identity(3, float)
            gvecs = mod_rvecs
            nvec = 0
        else:
            if not rvecs.ndim==2 or not rvecs.flags["C_CONTIGUOUS"] or rvecs.shape[0] > 3 or rvecs.shape[1] != 3:
                raise TypeError("rvecs must be a C contiguous array with three columns and at most three rows.")
            nvec = len(rvecs)
            Up, Sp, Vt = np.linalg.svd(rvecs, full_matrices=True)
            S = np.ones(3, float)
            S[:nvec] = Sp
            U = np.identity(3, float)
            U[:nvec,:nvec] = Up
            mod_rvecs = np.dot(U*S, Vt)
            mod_rvecs[:nvec] = rvecs
            gvecs = np.dot(U/S, Vt)
        cell.cell_update(self._c_domain, <double*>mod_rvecs.data, <double*>gvecs.data, nvec)

    def _get_nvec(self):
        """The number of domain vectors."""
        return cell.cell_get_nvec(self._c_domain)

    nvec = property(_get_nvec)

    def _get_volume(self):
        """The generalized volume of the unit domain (length, area or volume)."""
        return cell.cell_get_volume(self._c_domain)

    volume = property(_get_volume)

    def _get_rvecs(self, full=False):
        """The real-space domain vectors, layed out as rows."""
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_rvecs(self._c_domain, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rvecs = property(_get_rvecs)

    def _get_gvecs(self, full=False):
        """The reciprocal-space domain vectors, layed out as rows."""
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        cell.cell_copy_gvecs(self._c_domain, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gvecs = property(_get_gvecs)

    def _get_rspacings(self, full=False):
        """The (orthogonal) spacing between opposite sides of the real-space unit domain."""
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_rspacings(self._c_domain, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rspacings = property(_get_rspacings)

    def _get_gspacings(self, full=False):
        """The (orthogonal) spacing between opposite sides of the reciprocal-space unit domain."""
        cdef np.ndarray[double, ndim=1] result
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        cell.cell_copy_gspacings(self._c_domain, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gspacings = property(_get_gspacings)

    def _get_parameters(self):
        """The domain parameters (lengths and angles)."""
        rvecs = self._get_rvecs()
        tmp = np.dot(rvecs, rvecs.T)
        lengths = np.sqrt(np.diag(tmp))
        tmp /= lengths
        tmp /= lengths.reshape((-1,1))
        if len(rvecs) < 2:
            cosines = np.array([])
        elif len(rvecs) == 2:
            cosines = np.array([tmp[0,1]])
        else:
            cosines = np.array([tmp[1,2], tmp[2,0], tmp[0,1]])
        angles = np.arccos(np.clip(cosines, -1, 1))
        return lengths, angles

    parameters = property(_get_parameters)

    def mic(self, np.ndarray[double, ndim=1] delta):
        """Apply the minimum image convention to delta in-place."""
        assert delta.size == 3
        cell.cell_mic(<double*> delta.data, self._c_domain)

    def to_center(self, np.ndarray[double, ndim=1] pos):
        """Return the corresponding position in the central domain."""
        assert pos.size == 3
        cdef np.ndarray[long, ndim=1] result
        result = np.zeros(self.nvec, int)
        cell.cell_to_center(<double*> pos.data, self._c_domain, <long*> result.data)
        return result

    def add_vec(self, np.ndarray[double, ndim=1] delta, np.ndarray[long, ndim=1] r):
        """Add a linear combination of cell vectors, `r`, to `delta` in-place."""
        assert delta.size == 3
        assert r.size == self.nvec
        cell.cell_add_vec(<double*> delta.data, self._c_domain, <long*> r.data)


