import numpy as np

from opus import write_opus, parse_traj as parse_opus
from rruff import write_rruff, parse as parse_rruff
from spc import parse_traj as parse_spc
try:
  from renishaw import parse_wxd
except ImportError:
  print 'WXD parsing disabled until metakit is installed'
  def parse_wxd(f):
    raise NotImplementedError('WXD parsing relies on metakit')


def parse_loose(fh):
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
  next(fh)
  next(fh)
  data = np.loadtxt(fh)
  if not data.size or data.ndim != 2 or data.shape[1] != 3:
    raise ValueError('Invalid file for USGS ASC format')
  bands,intensities,stdv = data.T
  mask = (stdv >= 0) & (intensities > -1e3)  # exclude bogus negative values
  bands = bands[mask]
  intensities = intensities[mask]
  # convert microns to wavenumbers, reversing the order to keep it ascending
  bands = 10000./bands[::-1]
  return np.column_stack((bands, intensities))


PARSERS = {
    'opus': parse_opus,
    'spc': parse_spc,
    'wxd': parse_wxd,
    'rruff': parse_rruff,
    'asc': parse_asc,
    'txt': parse_loose,
}


def parse_spectrum(fh, filetype=None):
  '''Tries to parse a spectrum from an arbitrary file/filename.'''
  if not hasattr(fh, 'read'):
    return parse_spectrum(open(fh), filetype=filetype)
  # Use the specified parser
  if filetype is not None:
    return PARSERS[filetype](fh)
  # No parser specified, so let's try them all!
  # Try binary formats first, because they fail fast (magic number checks).
  for key in ('opus', 'spc', 'wxd', 'rruff', 'asc'):
    try:
      return PARSERS[key](fh)
    except:
      fh.seek(0)
  # Nothing worked, let's try a looser parse.
  return parse_loose(fh)
