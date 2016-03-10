import numpy as np

# Mask for LIBS channels
ALAMOS_MASK = np.zeros(6144, dtype=bool)
ALAMOS_MASK[110:1994] = True
ALAMOS_MASK[2169:4096] = True
ALAMOS_MASK[4182:5856] = True

# NOTE: the Ishikawa paper had a mistake!
# They put Annite in the Plagioclase group,
# but it should really be in the Mica group.
ISHIKAWA_MINERALS = set([
    # Plagioclase
    'Albite','Andesine','Annite','Anorthite',
    'Bytownite','Labradorite','Oligoclase',
    # Pyroxene
    'Augite','Clinoenstatite','Diopside','Enstatite',
    'Ferrosilite','Hedenbergite','Jadeite','Spodumene',
    # K-Spar
    'Anorthoclase','Microcline','Orthoclase',
    # Mica
    'Lepidolite','Muscovite','Phlogopite','Trilithionite','Zinnwaldite',
    # Olivine
    'Fayalite','Forsterite',
    # Quartz
    'Quartz'
])

def resample(spectrum, target_bands):
  m = spectrum[:,1].min()
  return np.interp(target_bands, spectrum[:,0], spectrum[:,1], left=m, right=m)

