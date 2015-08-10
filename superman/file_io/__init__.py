import numpy as np

from opus import write_opus, parse_traj as parse_opus
from rruff import parse as parse_rruff
from spc import parse_traj as parse_spc


def parse_spectrum(fh):
  '''Tries to parse a spectrum from an arbitrary file/filename.'''
  if not hasattr(fh, 'read'):
    return parse_spectrum(open(fh))
  # Try opus first, because it fails fast at a magic number check.
  for parse_fn in (parse_opus, parse_spc, parse_rruff):
    try:
      return parse_fn(fh)
    except:
      fh.seek(0)
  # Nothing worked, let's try a looser parse.
  data = np.atleast_2d(np.loadtxt(fh, dtype=np.float32))
  if data.shape[1] == 2 and data.shape[0] > 2:
    return data
  if data.shape[0] == 2 and data.shape[1] > 2:
    return data.T
  raise ValueError('Invalid shape for spectrum data: %s' % data.shape)


def parse_asc(fh):
  # Parser for USGS ascii files
  for line in fh:
    if line.startswith('------'):
      break
  # Skip two lines of header (XXX)
  foo = next(fh)
  bar = next(fh)
  bands,intensities,stdv = np.loadtxt(fh).T
  mask = (stdv >= 0) & (intensities > -1e3)  # exclude bogus negative values
  bands = bands[mask]
  intensities = intensities[mask]
  # convert microns to wavenumbers, reversing the order to keep it ascending
  bands = 10000./bands[::-1]
  return np.column_stack((bands, intensities))

