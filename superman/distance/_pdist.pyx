# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
from cython.parallel import prange
from libc.math cimport fmin, fabs, sqrt

cpdef void match_between(double[:,::1] query, double[:,::1] target,
                         double exp, double[:,::1] dist) nogil:
  cdef Py_ssize_t n1 = query.shape[0], n2 = target.shape[0], d = query.shape[1]
  cdef Py_ssize_t i, j, k
  cdef double score
  # For all pairs of spectra
  for i in prange(n1, nogil=True):
    for j in range(n2):
      dist[i,j] = score_match(query[i], target[j], exp, d)


cpdef void match_within(double[:,::1] spectra, double exp,
                        double[:,::1] dist) nogil:
  cdef Py_ssize_t n = spectra.shape[0], d = spectra.shape[1]
  cdef Py_ssize_t i, j, k
  cdef double score
  # Upper-right triangle only
  for i in prange(n-1, nogil=True):
    for j in range(i+1, n):
      score = score_match(spectra[i], spectra[j], exp, d)
      dist[i,j] = score
      dist[j,i] = score


cpdef void combo_between(double[:,::1] query, double[:,::1] target,
                         double w, double[:,::1] dist) nogil:
  cdef Py_ssize_t n1 = query.shape[0], n2 = target.shape[0], d = query.shape[1]
  cdef Py_ssize_t i, j
  # For all pairs of spectra
  for i in prange(n1, nogil=True):
    for j in range(n2):
      dist[i,j] = score_combo(query[i], target[j], w, d)


cpdef void combo_within(double[:,::1] spectra, double w,
                        double[:,::1] dist) nogil:
  cdef Py_ssize_t n = spectra.shape[0], d = spectra.shape[1]
  cdef Py_ssize_t i, j
  cdef double score
  # Upper-right triangle only
  for i in prange(n-1, nogil=True):
    for j in range(i+1, n):
      score = score_combo(spectra[i], spectra[j], w, d)
      dist[i,j] = score
      dist[j,i] = score


cdef inline double score_match(double[::1] spec1, double[::1] spec2,
                               double exp, Py_ssize_t n) nogil:
  cdef double score = 0.0, ssa = 0.0, ssb = 0.0, a, b
  cdef Py_ssize_t k
  for k in range(n):
    a = spec1[k]
    b = spec2[k]
    score += ms(a, b, exp)
    ssa += a * a
    ssb += b * b
  return score / sqrt(ssa * ssb) + 1

cdef inline double score_combo(double[::1] spec1, double[::1] spec2,
                               double w, Py_ssize_t n) nogil:
  cdef double score = 0.0, ssa = 0.0, ssb = 0.0, a, b
  cdef Py_ssize_t k
  for k in range(n):
    a = spec1[k]
    b = spec2[k]
    score += combo(a, b, w)
    ssa += a * a
    ssb += b * b
  return score / sqrt(ssa * ssb) + 1

cdef inline double ms(double ay, double by, double exp) nogil:
  return fmin(ay, by) * (1.0 - fabs(ay - by) ** exp)

cdef inline double combo(double ay, double by, double w) nogil:
  return w * fabs(ay - by) - (1 - w) * (ay * by)


cpdef Py_ssize_t score_pdist(char[:,::1] dana_dist, double[:,::1] test_dist) nogil:
  cdef Py_ssize_t i, num_wrong = 0, n = dana_dist.shape[0]
  # For all rows:
  for i in prange(n, nogil=True):
    num_wrong += score_pdist_row(dana_dist[i], test_dist[i], i, n)
  return num_wrong


cpdef Py_ssize_t score_pdist_row(char[::1] dana, double[::1] test,
                                 Py_ssize_t i, Py_ssize_t n) nogil:
  cdef Py_ssize_t j, k_start, k, num_wrong = 0
  # For the upper trianglular columns
  k_start = i + 2
  for j in range(i+1, n):
    # If dana[j] is 5, no dana[k] will be greater.
    if dana[j] == 5:
      break
    # Search for the next k where dana[k] > dana[j]
    while dana[j] == dana[k_start]:
      k_start += 1
    # Count violations (where test[j] >= test[k])
    for k in range(k_start, n):
      if (test[j] >= test[k]):
        num_wrong += 1
  return num_wrong