# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython
from cython cimport floating
from libc.math cimport fmin, fabs, sqrt

IDX_DTYPE = np.int32
ctypedef np.int32_t IDX_DTYPE_t


cpdef float traj_match(floating[:,::1] a, floating[:,::1] b,
                       float intensity_eps) nogil:
  cdef IDX_DTYPE_t na = <IDX_DTYPE_t> a.shape[0]+1
  cdef IDX_DTYPE_t nb = <IDX_DTYPE_t> b.shape[0]+1
  cdef IDX_DTYPE_t i, j, min_j = 1
  cdef float dx, ay, by, lcs_diag = 0, ssa = 0, ssb = 0
  cdef floating[::1] aa, bb
  cdef float band_eps = optimal_band_eps(a, b)
  for i in range(na-1):
    aa = a[i,:]
    for j in range(min_j, nb):
      bb = b[j-1,:]
      dx = aa[0] - bb[0]
      if fabs(dx) <= band_eps:
        ay = aa[1]
        by = bb[1]
        lcs_diag += match_score(ay, by, intensity_eps)
        ssa += ay*ay
        ssb += by*by
      elif dx > 0:
        # b is still behind a
        min_j = j
      else:
        # b is past a, so no more matches, skip ahead
        break
  return lcs_diag / sqrt(ssa * ssb)


cpdef float traj_combo(floating[:,::1] a, floating[:,::1] b, float w) nogil:
  cdef IDX_DTYPE_t na = <IDX_DTYPE_t> a.shape[0]+1
  cdef IDX_DTYPE_t nb = <IDX_DTYPE_t> b.shape[0]+1
  cdef IDX_DTYPE_t i, j, min_j = 1
  cdef float dx, ay, by, lcs_diag = 0, ssa = 0, ssb = 0
  cdef floating[::1] aa, bb
  cdef float band_eps = optimal_band_eps(a, b)
  for i in range(na-1):
    aa = a[i,:]
    for j in range(min_j, nb):
      bb = b[j-1,:]
      dx = aa[0] - bb[0]
      if fabs(dx) <= band_eps:
        ay = aa[1]
        by = bb[1]
        lcs_diag += combo_score(ay, by, w)
        ssa += ay*ay
        ssb += by*by
      elif dx > 0:
        # b is still behind a
        min_j = j
      else:
        # b is past a, so no more matches, skip ahead
        break
  return lcs_diag / sqrt(ssa * ssb)


cpdef float optimal_band_eps(floating[:,::1] a,
                             floating[:,::1] b) nogil:
  cdef float adiff = mean_band_difference(a)
  cdef float bdiff = mean_band_difference(b)
  return sqrt(adiff * bdiff) / 2.0  # half the geometric mean


# Doesn't operate directly on a floating[:] x because of fused type mess.
cdef float mean_band_difference(floating[:,::1] x) nogil:
  cdef float running_sum = 0
  cdef IDX_DTYPE_t i, n = <IDX_DTYPE_t> x.shape[0] - 1
  for i in range(n):
    running_sum += x[i+1,0] - x[i,0]
  return running_sum / n


cdef inline float match_score(float ay, float by, float intensity_eps) nogil:
  return fmin(ay, by) * (1.0 - fabs(ay - by) ** intensity_eps)


cdef inline float combo_score(float ay, float by, float w) nogil:
  return w * fabs(ay - by) - (1 - w) * (ay * by)
