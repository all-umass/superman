import numpy as np

from ishikawa import filter_ishikawa, filter_ishikawa_traj

# Mask for LIBS channels
ALAMOS_MASK = np.zeros(6144, dtype=bool)
ALAMOS_MASK[110:1994] = True
ALAMOS_MASK[2169:4096] = True
ALAMOS_MASK[4182:5856] = True


def resample(spectrum, target_bands):
  m = spectrum[:,1].min()
  return np.interp(target_bands, spectrum[:,0], spectrum[:,1], left=m, right=m)


def prepare_data(data_file, opts):
  npz = np.load(data_file)
  lasers = npz['lasers']
  Y = npz['labels']
  names = npz['names']
  traj = 'data' not in npz.files
  if not traj:
    X = npz['data']
  for laser in opts.laser:
    mask = Ellipsis if laser == 'all' else lasers == laser
    names_ = names[mask]
    Y_ = Y[mask]
    if traj:
      X_ = [npz[k] for k in names_]
    else:
      X_ = X[mask]
    yield _post_prepare(X_, Y_, names_, traj, opts)


def _post_prepare(X, Y, names, traj, opts):
  if opts.ishikawa:
    filter_fn = filter_ishikawa_traj if traj else filter_ishikawa
    X, Y, names = filter_fn(X, Y, names)
  labels, label_map = _convert_labels(Y, names)
  return X, labels, label_map, names


def _convert_labels(Y, names):
  _, class_idxs, labels = np.unique(Y, return_index=True,
                                    return_inverse=True)
  label_map = np.array([n.split('-',1)[0] for n in names[class_idxs]])
  return labels, label_map
