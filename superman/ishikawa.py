
# NOTE: the Ishikawa paper had a mistake!
# They put Annite in the Plagioclase group,
# but it should really be in the Mica group.

ishikawa_minerals = set([
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

ishikawa_groups = {
    'PLA': set(['Albite','Andesine','Annite','Anorthite',
                'Bytownite','Labradorite','Oligoclase']),
    'PYX': set(['Augite','Clinoenstatite','Diopside','Enstatite',
                'Ferrosilite','Hedenbergite','Jadeite','Spodumene']),
    'KSP': set(['Anorthoclase','Microcline','Orthoclase']),
    'MCA': set(['Lepidolite','Muscovite','Phlogopite',
                'Trilithionite','Zinnwaldite']),
    'OLI': set(['Fayalite','Forsterite']),
    'QTZ': set(['Quartz'])
}


def _ishi_indices(names):
  return [i for i,n in enumerate(names)
          if n.split('-',1)[0] in ishikawa_minerals]


def filter_ishikawa(X, Y, names):
  ishi_idx = _ishi_indices(names)
  return X[ishi_idx], Y[ishi_idx], names[ishi_idx]


def filter_ishikawa_traj(traj, Y, names):
  ishi_idx = _ishi_indices(names)
  ishi_traj = [traj[i] for i in ishi_idx]
  return ishi_traj, Y[ishi_idx], names[ishi_idx]
