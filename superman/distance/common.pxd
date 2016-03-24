# -*- cython -*-
cpdef float optimal_band_eps(float[:,::1] a, float[:,::1] b) nogil
cdef float match_score(float ay, float by, float intensity_eps) nogil
cdef float combo_score(float ay, float by, float w) nogil
cdef float minimum(float[::1] buf, Py_ssize_t n) nogil
