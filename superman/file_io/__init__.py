from __future__ import absolute_import, print_function
import numpy as np
import os
import warnings

try:
  import xylib
  HAS_XYLIB = True
except ImportError:
  warnings.warn('xylib not found; some formats are unavailable.')
  HAS_XYLIB = False

try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  warnings.warn('pandas not found; excel parsing is unavailable.')
  HAS_PANDAS = False

from .opus import write_opus, parse_traj as parse_opus
from .rruff import write_rruff, parse as parse_rruff
from .spc import parse_traj as parse_spc
from .andor import parse_sif
from .bruker_raw import parse_raw
try:
  from .renishaw import parse_wxd
  HAS_WXD = True
except ImportError:
  warnings.warn('metakit not found; WXD parsing is unavailable.')
  HAS_WXD = False


def parse_loose(fh):
  try:
    data = np.loadtxt(fh, dtype=np.float32, usecols=(0,1))
  except (ValueError, IndexError):
    # default parse failed, try parsing as CSV
    fh.seek(0)
    data = np.loadtxt(fh, dtype=np.float32, delimiter=',', usecols=(0,1))
  return spectrum_shaped(data)


def spectrum_shaped(data):
  data = np.atleast_2d(data)
  if data.shape[1] in (2, 3) and data.shape[0] > 2:
    return data[:,:2]
  if data.shape[0] == 2 and data.shape[1] > 2:
    return data.T
  raise ValueError('Invalid shape for spectrum data: %s' % (data.shape,))


def parse_asc(fh):
  # Parser for USGS ascii files
  for line in fh:
    if line.startswith('------'):
      break
  # Skip two lines of header (XXX)
  next(fh)  # title
  next(fh)  # history
  data = np.loadtxt(fh)
  # 3 columns: wave, ints, stdv
  if not data.size or data.ndim != 2 or data.shape[1] != 3:
    raise ValueError('Invalid file for USGS ASC format')
  # exclude missing values, reported as large negatives
  data = data[data.min(axis=1) > -1e10]
  # convert microns to nanometers
  data[:,0] *= 1000
  return data[:,:2]


def parse_bwspec(fh):
  # Parser for BWSpec4 ascii files
  version = next(fh)
  if not version.startswith('File Version;BWSpec4.'):
    raise ValueError('Invalid file for BWSpec4.xx format')
  for line in fh:
    if line.startswith('Pixel;'):
      break
  else:
    raise ValueError('No data for BWSpec4.xx format')
  colnames = line.split(';')
  data = np.genfromtxt(fh, delimiter=';', loose=True, invalid_raise=False)
  # TODO: also may have Wavenumber, Raman Shift
  xidx = colnames.index('Wavelength')
  try:
    yidx = colnames.index('Dark Subtracted #1')
  except ValueError:
    yidx = colnames.index('Raw data #1')
  return data[:, [xidx, yidx]]


def parse_xlsx(fh):
  df = pd.read_excel(fh)
  return spectrum_shaped(df.values)


PARSERS = {
    'opus': parse_opus,
    'spc': parse_spc,
    'raw': parse_raw,
    'sif': parse_sif,
    'bwspec': parse_bwspec,
    'rruff': parse_rruff,
    'asc': parse_asc,
    'txt': parse_loose,
}
if HAS_WXD:
  PARSERS['wxd'] = parse_wxd
if HAS_PANDAS:
  PARSERS['xlsx'] = parse_xlsx

# Try binary formats first, because they fail fast (magic number checks).
PARSE_ORDER = [PARSERS[k] for k in (
    'opus', 'spc', 'raw', 'wxd', 'sif', 'xlsx', 'bwspec', 'rruff', 'asc'
    ) if k in PARSERS]


def _parse_with_xylib(filepath):
  data = xylib.load_file(filepath)
  num_blocks = data.get_block_count()
  if num_blocks != 1:
    raise ValueError('expected only one block, got %d' % num_blocks)

  block = data.get_block(0)
  num_cols = block.get_column_count()
  if num_cols not in (1, 2):
    raise ValueError('expected 1 or 2 columns, got %d' % num_cols)

  num_pts = block.get_point_count()
  traj = np.empty((num_pts, 2), dtype=np.float32)
  if num_cols == 1:
    col = block.get_column(1)
    for i in range(num_pts):
      traj[i,:] = (i, col.get_value(i))
  else:
    xcol = block.get_column(1)
    ycol = block.get_column(2)
    for i in range(num_pts):
      traj[i,:] = (xcol.get_value(i), ycol.get_value(i))
  return traj


def parse_spectrum(fh, filetype=None):
  '''Tries to parse a spectrum from an arbitrary file-like/path.

  fh : file-like object or str, input file to parse
  filetype : str, should be a key in the PARSERS dict

  Returns a trajectory: (n,2)-array of float32
  '''
  if hasattr(fh, 'read'):
    fileobj = fh
    filepath = getattr(fh, 'name', '')
  else:
    filepath = fh
    fileobj = open(filepath, 'rU')

  # Use the specified parser, if given
  if filetype is not None:
    return PARSERS[filetype](fileobj)

  # Try to use xylib (can't handle file-like objects, sadly)
  if HAS_XYLIB and filepath and os.path.exists(filepath):
    try:
      return _parse_with_xylib(filepath)
    except:
      pass

  # No parser specified, so let's try them all!
  for parser in PARSE_ORDER:
    try:
      return parser(fileobj)
    except:
      fileobj.seek(0)
  # Nothing worked, let's try a looser parse.
  return parse_loose(fileobj)


def write_spectrum(filename, traj, filetype='txt', comments=''):
  '''Writes a spectrum to file with the specified filetype.
  Filetype choices are: ('txt', 'rruff', 'opus')
  '''
  if filetype in ('rruff', 'txt'):
    return write_rruff(filename, traj, comments)
  if filetype == 'opus':
    return write_opus(filename, traj, comments)
  raise ValueError('Unknown filetype for writing: %r' % filetype)
