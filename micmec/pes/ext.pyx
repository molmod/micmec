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

#cython: embedsignature=True

"""C routines

This extension module is used by various modules of the ``micmec.pes`` package.
"""


import numpy as np
cimport numpy as np
cimport domain

from micmec.log import log


__all__ = ["Domain"]


cdef class Domain:
    """Representation of periodic boundary conditions."""
    cdef domain.domain_type* _c_domain

    def __cinit__(self):
        self._c_domain = domain.domain_new()
        if self._c_domain is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_domain is not NULL:
            domain.domain_free(self._c_domain)

    def __init__(self, np.ndarray[double, ndim=2] rvecs):
        self.update_rvecs(rvecs)

    def update_rvecs(self, np.ndarray[double, ndim=2] rvecs):
        cdef np.ndarray[double, ndim=2] mod_rvecs
        cdef np.ndarray[double, ndim=2] gvecs
        cdef int nvec
        if rvecs is None or rvecs.size == 0:
            mod_rvecs = np.identity(3, float)
            gvecs = mod_rvecs
            nvec = 0
        else:
            if not rvecs.ndim==2 or not rvecs.flags["C_CONTIGUOUS"] or rvecs.shape[0] > 3 or rvecs.shape[1] != 3:
                raise TypeError("rvecs must be a C-contiguous array with three columns and at most three rows.")
            nvec = len(rvecs)
            Up, Sp, Vt = np.linalg.svd(rvecs, full_matrices=True)
            S = np.ones(3, float)
            S[:nvec] = Sp
            U = np.identity(3, float)
            U[:nvec,:nvec] = Up
            mod_rvecs = np.dot(U*S, Vt)
            mod_rvecs[:nvec] = rvecs
            gvecs = np.dot(U/S, Vt)
        domain.domain_update(self._c_domain, <double*>mod_rvecs.data, <double*>gvecs.data, nvec)

    def _get_nvec(self):
        return domain.domain_get_nvec(self._c_domain)

    nvec = property(_get_nvec)

    def _get_volume(self):
        return domain.domain_get_volume(self._c_domain)

    volume = property(_get_volume)

    def _get_rvecs(self, full=False):
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        domain.domain_copy_rvecs(self._c_domain, <double*>result.data, full)
        result.setflags(write=False)
        return result

    rvecs = property(_get_rvecs)

    def _get_gvecs(self, full=False):
        cdef np.ndarray[double, ndim=2] result
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        domain.domain_copy_gvecs(self._c_domain, <double*>result.data, full)
        result.setflags(write=False)
        return result

    gvecs = property(_get_gvecs)

    def _get_parameters(self):
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

