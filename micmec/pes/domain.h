#ifndef MICMEC_PES_DOMAIN_H
#define MICMEC_PES_DOMAIN_H

typedef struct {
  double rvecs[9], gvecs[9];
  double volume;
  int nvec;
} domain_type;

domain_type* domain_new(void);
void domain_free(domain_type* domain);
void domain_update(domain_type* domain, double *rvecs, double *gvecs, int nvec);

int domain_get_nvec(domain_type* domain);
double domain_get_volume(domain_type* domain);
void domain_copy_rvecs(domain_type* domain, double *rvecs, int full);
void domain_copy_gvecs(domain_type* domain, double *gvecs, int full);

#endif

