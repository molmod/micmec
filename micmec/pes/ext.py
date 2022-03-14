#!/usr/bin/env python

class Domain:
    """
    Representation of periodic boundary conditions.
    0, 1, 2 and 3 dimensional systems are supported. 
    The domain vectors need not to be orthogonal.
    """
    
    def __init__(self, *args, **kwargs):
        self._domain = domain.domain_new()
        if self._domain is None:
            raise MemoryError()

    
    def __init__(self, np.ndarray[double, ndim=2] rvecs):
        """
        **ARGUMENTS**
        rvecs
            A numpy array with at most three domain vectors, layed out as
            rows in a rank-2 matrix. For non-periodic systems, this array
            must have shape (0,3).
        """
        self.update_rvecs(rvecs)


    def update_rvecs(self, rvecs):
        """
        Change the domain vectors and recompute the reciprocal domain vectors.

        **ARGUMENTS**
        rvecs
            A numpy array with at most three domain vectors, layed out as
            rows in a rank-2 matrix. For non-periodic systems, this array
            must have shape (0,3).
        """
        if rvecs is None or rvecs.size == 0:
            mod_rvecs = np.identity(3, float)
            gvecs = mod_rvecs
            nvec = 0
        else:
            if not rvecs.ndim == 2 or rvecs.shape[0] > 3 or rvecs.shape[1] != 3:
                raise TypeError("rvecs must be an array with three columns and at most three rows.")
            nvec = len(rvecs)
            Up, Sp, Vt = np.linalg.svd(rvecs, full_matrices=True)
            S = np.ones(3, float)
            S[:nvec] = Sp
            U = np.identity(3, float)
            U[:nvec,:nvec] = Up
            mod_rvecs = np.dot(U*S, Vt)
            mod_rvecs[:nvec] = rvecs
            gvecs = np.dot(U/S, Vt)
        domain.domain_update(self._domain, <double*>mod_rvecs.data, <double*>gvecs.data, nvec)

    
    def _get_nvec(self):
        """The number of domain vectors."""
        return domain.domain_get_nvec(self._domain)

    nvec = property(_get_nvec)

    def _get_volume(self):
        """The generalized volume of the domain (length, area or volume)."""
        return domain.domain_get_volume(self._domain)

    volume = property(_get_volume)

    def _get_rvecs(self, full=False):
        """The real-space domain vectors, layed out as rows."""
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        domain.domain_copy_rvecs(self._domain, result.data, full)
        result.setflags(write=False)
        return result

    rvecs = property(_get_rvecs)

    def _get_gvecs(self, full=False):
        """The reciprocal-space domain vectors, layed out as rows."""
        if full:
            result = np.zeros((3, 3), float)
        else:
            result = np.zeros((self.nvec, 3), float)
        domain.domain_copy_gvecs(self._domain, result.data, full)
        result.setflags(write=False)
        return result

    gvecs = property(_get_gvecs)

    def _get_rspacings(self, full=False):
        """The (orthogonal) spacing between opposite sides of the real-space unit domain."""
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        domain.domain_copy_rspacings(self._domain, result.data, full)
        result.setflags(write=False)
        return result

    rspacings = property(_get_rspacings)

    def _get_gspacings(self, full=False):
        """The (orthogonal) spacing between opposite sides of the reciprocal-space unit domain."""
        if full:
            result = np.zeros(3, float)
        else:
            result = np.zeros(self.nvec, float)
        domain.domain_copy_gspacings(self._domain, result.data, full)
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

    
    def mic(self, delta):
        """Apply the minimum image convention to delta in-place."""
        assert delta.size == 3
        domain.domain_mic(delta.data, self._domain)

    
    def to_center(self, pos):
        """Return the corresponding position in the central domain."""
        assert pos.size == 3
        result = np.zeros(self.nvec, int)
        domain.domain_to_center(pos.data, self._domain, result.data)
        return result
    

    def add_vec(self, delta, r):
        """Add a linear combination of domain vectors, `r`, to `delta` in-place."""
        assert delta.size == 3
        assert r.size == self.nvec
        domain.domain_add_vec(delta.data, self._domain, r.data)

    
    def compute_distances(self, output, pos0, pos1=None, pairs=None, do_include=False, nimage=0):
        """
        Computes all distances between the given coordinates.
        
        **ARGUMENTS**
        output
            An numpy vector of the proper length that will be used to store
            all the distances.
        pos0
            An array with Cartesian coordinates

        **OPTIONAL ARGUMENTS**
        pos1
            A second array with Cartesian coordinates
        pairs
            A sorted array of atom pairs. When do_include==False, this list
            will be excluded from the computation. When do_include==True,
            only these pairs are considered when computing distances.
            The indexes in this array refer to rows of pos0 or pos1. If pos1
            is not given, both columns refer to rows of pos0. If pos1 is
            given, the first column refers to rows of pos0 and the second
            column refers to rows of pos1. The rows in the pairst array
            should be sorted lexicographically, first along the first
            column, then along the second column.
        do_include
            True or False, controls how the pairs list is interpreted. When
            set to True, nimage must be zero and the pairs attribute must be
            a non-empty array.
        nimage
            The number of domain images to consider in the computation of the
            pair distances. By default, this is zero, meaning that only the
            minimum image convention is used.

        This routine can operate in two different ways, depending on the presence/absence 
        of the argument `pos1`. If not given, all distances between points in `pos0` 
        are computed and the length of the output array is `len(pos0)*(len(pos0)-1)/2`. 
        If `pos1` is given, all distances are computed between a point in `pos0` and a
        point in `pos1` and the length of the output array is `len(pos0)*len(pos1)`.
        In both cases, some pairs of atoms may be excluded from the output with the 
        `exclude` argument. In typical cases, this list of excluded pairs is relatively 
        short. In case, the exclude argument is present the number of computed distances
        is less than explained above, but it is recommended to still use those sizes 
        in case some pairs in the excluded list are not applicable.
        """
        assert pos0.shape[1] == 3
        assert nimage >= 0
        natom0 = pos0.shape[0]

        if pairs is not None:
            assert pairs.shape[1] == 2
            pairs_pointer = pairs.data
            npair = pairs.shape[0]
        else:
            pairs_pointer = None
            npair = 0

        if nimage > 0:
            if self.nvec == 0:
                raise ValueError("Can only include distances to periodic images for periodic systems.")
            factor = (1+2*nimage)**self.nvec
        else:
            factor = 1

        if do_include:
            if nimage != 0:
                raise ValueError("When do_include==True, nimage must be zero.")
            if npair == 0:
                raise ValueError("No pairs given and do_include==True.")

        if pos1 is None:
            if do_include:
                npair == output.shape[0]
            else:
                assert factor*(natom0*(natom0-1))//2 - npair == output.shape[0]
            if domain.is_invalid_exclude(pairs_pointer, natom0, natom0, npair, True):
                raise ValueError("The pairs array must countain indices within proper bounds and must be lexicographically sorted.")
            domain.domain_compute_distances1(self._domain, pos0.data, output.data, 
                                            natom0, pairs_pointer, npair, do_include, nimage)
        else:
            assert pos1.shape[1] == 3
            natom1 = pos1.shape[0]

            if do_include:
                npair == output.shape[0]
            else:
                assert factor*natom0*natom1 - npair == output.shape[0]
            if domain.is_invalid_exclude(pairs_pointer, natom0, natom1, npair, False):
                raise ValueError("The pairs array must countain indices within proper bounds and must be lexicographically sorted.")
            domain.domain_compute_distances2(self._domain, pos0.data, pos1.data, output.data, 
                                            natom0, natom1, pairs_pointer, npair, do_include, nimage)



