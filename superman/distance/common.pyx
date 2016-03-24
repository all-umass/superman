# -*- cython -*-

# cython: initializedcheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
from libc.math cimport fmin, fabs, sqrt


cpdef float optimal_band_eps(float[:,::1] a, float[:,::1] b) nogil:
  cdef float adiff = mean_band_difference(a)
  cdef float bdiff = mean_band_difference(b)
  return sqrt(adiff * bdiff) / 2.0  # half the geometric mean


cdef float mean_band_difference(float[:,::1] x) nogil:
  cdef float running_sum = 0
  cdef Py_ssize_t i, n = x.shape[0]
  # work with the raw pointer for speed, because we know the shape is (n,2)
  cdef float* xx = &x[0,0]
  for i in range(n-1):
    running_sum += xx[2*i+2] - xx[2*i]
  return running_sum / (n-1)


cdef inline float match_score(float ay, float by, float intensity_eps) nogil:
  return fmin(ay, by) * (1.0 - fabs(ay - by) ** intensity_eps)


cdef inline float combo_score(float ay, float by, float w) nogil:
  return w * fabs(ay - by) - (1 - w) * (ay * by)


cdef inline float minimum(float[::1] buf, Py_ssize_t n) nogil:
  cdef float x, minval = buf[0]
  cdef Py_ssize_t i
  for i in range(1, n):
    x = buf[i]
    if x < minval:
      minval = x
  return minval

