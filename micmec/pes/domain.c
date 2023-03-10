#include <stdlib.h>
#include <math.h>
#include "domain.h"

domain_type* domain_new(void) {
  return malloc(sizeof(domain_type));
}

void domain_free(domain_type* domain) {
  free(domain);
}

void domain_update(domain_type* domain, double *rvecs, double *gvecs, int nvec) {
  double tmp;
  int i;
  // copy everything
  (*domain).nvec = nvec;
  for (i=0; i<9; i++) {
    (*domain).rvecs[i] = rvecs[i];
    (*domain).gvecs[i] = gvecs[i];
  }
  // compute the volume
  switch(nvec) {
    case 0:
      (*domain).volume = 0.0;
      break;
    case 1:
      (*domain).volume = sqrt(
        rvecs[0]*rvecs[0]+rvecs[1]*rvecs[1]+rvecs[2]*rvecs[2]
      );
      break;
    case 2:
      tmp = rvecs[0]*rvecs[3]+rvecs[1]*rvecs[4]+rvecs[2]*rvecs[5];
      tmp = (rvecs[0]*rvecs[0]+rvecs[1]*rvecs[1]+rvecs[2]*rvecs[2])*
            (rvecs[3]*rvecs[3]+rvecs[4]*rvecs[4]+rvecs[5]*rvecs[5]) - tmp*tmp;
      if (tmp > 0) {
        (*domain).volume = sqrt(tmp);
      } else {
        (*domain).volume = 0.0;
      }
      break;
    case 3:
      (*domain).volume = fabs(
        rvecs[0]*(rvecs[4]*rvecs[8]-rvecs[5]*rvecs[7])+
        rvecs[1]*(rvecs[5]*rvecs[6]-rvecs[3]*rvecs[8])+
        rvecs[2]*(rvecs[3]*rvecs[7]-rvecs[4]*rvecs[6])
      );
      break;
  }
}


int domain_get_nvec(domain_type* domain) {
  return (*domain).nvec;
}

double domain_get_volume(domain_type* domain) {
  return (*domain).volume;
}

void domain_copy_rvecs(domain_type* domain, double *rvecs, int full) {
  int i, n;
  n = (full)?9:(*domain).nvec*3;
  for (i=0; i<n; i++) rvecs[i] = (*domain).rvecs[i];
}

void domain_copy_gvecs(domain_type* domain, double *gvecs, int full) {
  int i, n;
  n = (full)?9:(*domain).nvec*3;
  for (i=0; i<n; i++) gvecs[i] = (*domain).gvecs[i];
}

