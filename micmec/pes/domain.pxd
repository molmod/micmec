cdef extern from "domain.h":
    ctypedef struct domain_type:
        pass

    domain_type* domain_new()
    void domain_free(domain_type* domain)
    void domain_update(domain_type* domain, double *rvecs, double *gvecs, int nvec)

    int domain_get_nvec(domain_type* domain)
    double domain_get_volume(domain_type* domain)
    void domain_copy_rvecs(domain_type* domain, double *rvecs, bint full)
    void domain_copy_gvecs(domain_type* domain, double *gvecs, bint full)

