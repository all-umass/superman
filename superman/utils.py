import numpy as np

# Mask for LIBS channels
ALAMOS_MASK = np.zeros(6144, dtype=bool)
ALAMOS_MASK[110:1994] = True
ALAMOS_MASK[2169:4096] = True
ALAMOS_MASK[4182:5856] = True


def resample(spectrum, target_bands):
  m = spectrum[:,1].min()
  return np.interp(target_bands, spectrum[:,0], spectrum[:,1], left=m, right=m)
