from __future__ import absolute_import, print_function
import numpy as np
from datetime import datetime
from construct import (
    Tell, Array, BitStruct, Byte, ConstructError, Embedded, Flag, If, IfThenElse,
    Padding, Pointer, Bytes, Struct, Switch, Computed, this,
    Float64l, Float32l, Int32sl, Int16sl, Int32ul
)

from .construct_utils import BitSplitter, FixedSizeCString, LazyField

X_AXIS_LABELS = [
    "Arbitrary", "Wavenumber (cm-1)", "Micrometers (um)", "Nanometers (nm)",
    "Seconds ", "Minutes", "Hertz (Hz)", "Kilohertz (KHz)", "Megahertz (MHz) ",
    "Mass (M/z)", "Parts per million (PPM)", "Days", "Years",
    "Raman Shift (cm-1)", "eV",
    "XYZ text labels in fcatxt (old 0x4D version only)", "Diode Number",
    "Channel", "Degrees", "Temperature (F)", "Temperature (C)",
    "Temperature (K)", "Data Points", "Milliseconds (mSec)",
    "Microseconds (uSec) ", "Nanoseconds (nSec)", "Gigahertz (GHz)",
    "Centimeters (cm)", "Meters (m)", "Millimeters (mm)", "Hours"
]

Y_AXIS_LABELS = {
    0: "Arbitrary Intensity", 1: "Interferogram", 2: "Absorbance",
    3: "Kubelka-Munk", 4: "Counts", 5: "Volts", 6: "Degrees", 7: "Milliamps",
    8: "Millimeters", 9: "Millivolts", 10: "Log(1/R)", 11: "Percent",
    12: "Intensity", 13: "Relative Intensity", 14: "Energy", 16: "Decibel",
    19: "Temperature (F)", 20: "Temperature (C)", 21: "Temperature (K)",
    22: "Index of Refraction [N]", 23: "Extinction Coeff. [K]", 24: "Real",
    25: "Imaginary", 26: "Complex", 128: "Transmission", 129: "Reflectance",
    130: "Arbitrary or Single Beam with Valley Peaks", 131: "Emission"
}

EXPERIMENT_TYPES = [
    "General SPC", "Gas Chromatogram", "General Chromatogram",
    "HPLC Chromatogram", "FT-IR, FT-NIR, FT-Raman Spectrum or Igram",
    "NIR Spectrum", "UV-VIS Spectrum", "X-ray Diffraction Spectrum",
    "Mass Spectrum ", "NMR Spectrum or FID", "Raman Spectrum",
    "Fluorescence Spectrum", "Atomic Spectrum",
    "Chromatography Diode Array Spectra"
]

VERSIONS = {
    b'K': 'new LSB 1st',
    b'L': 'new MSB 1st',
    b'M': 'old format'
}

TFlags = BitStruct(
    'has_xs'/Flag,
    'use_subfile_xs'/Flag,
    'use_catxt_labels'/Flag,
    'ordered_subtimes'/Flag,
    'arbitrary_time'/Flag,
    'multiple_ys'/Flag,
    'enable_experiment'/Flag,
    'short_y'/Flag
)

DateVersionK = BitSplitter(
    Int32ul,
    minute=(0, 6),
    hour=(6, 5),
    day=(11, 5),
    month=(16, 4),
    year=(20, 12)
)

DateVersionM = Struct(
    'year1900'/Int16sl,  # XXX: guessing that we count years since 1900
    'year'/Computed(this.year1900 + 1900),
    'month'/Byte,
    'day'/Byte,
    'hour'/Byte,
    'minute'/Byte
)

HeaderVersionK = Struct(
    'experiment_type'/Byte,
    'exp'/Byte,
    'npts'/Int32sl,
    'first'/Float64l,
    'last'/Float64l,
    'nsub'/Int32sl,
    'xtype'/Byte,
    'ytype'/Byte,
    'ztype'/Byte,
    'post'/Byte,
    'date'/DateVersionK,
    Padding(9),
    'source'/FixedSizeCString(9),
    'peakpt'/Int16sl,
    Padding(32),
    'comment'/FixedSizeCString(130),
    'catxt'/FixedSizeCString(30),
    'log_offset'/Int32sl,
    'mods'/Int32sl,
    'procs'/Byte,
    'level'/Byte,
    'sampin'/Int16sl,
    'factor'/Float32l,
    'method'/FixedSizeCString(48),
    'zinc'/Float32l,
    'wplanes'/Int32sl,
    'winc'/Float32l,
    'wtype'/Byte,
    Padding(187)
)

HeaderVersionM = Struct(
    'exp'/Int16sl,
    'npts'/Float32l,
    'first'/Float32l,
    'last'/Float32l,
    'xtype'/Byte,
    'ytype'/Byte,
    'date'/DateVersionM,
    Padding(8),  # res
    'peakpt'/Int16sl,
    'nscans'/Int16sl,
    Padding(28),  # spare
    'comment'/FixedSizeCString(130),
    'catxt'/FixedSizeCString(30),
    # only one subfile supported by this version
    'nsub'/Computed(1),
    # log data is not supported by this version
    'log_offset'/Computed(0)
)


def _wrong_version_error(ctx):
  raise NotImplementedError('SPC version %s is not implemented' % ctx.version)

Header = Struct(
    'TFlags'/TFlags,
    'version'/Bytes(1),
    'header'/Switch(this.version, {
        b'K': HeaderVersionK,
        b'L': Computed(_wrong_version_error),
        b'M': HeaderVersionM,
    }, default=Computed(_wrong_version_error))
)

Subfile = Struct(
    'flags'/Byte,
    'exp'/Byte,
    'index'/Int16sl,
    'time'/Float32l,
    'next'/Float32l,
    'nois'/Float32l,
    'npts'/Int32sl,
    'scan'/Int32sl,
    'wlevel'/Float32l,
    Padding(4),
    'float_y'/Computed(this.exp == 128),
    'num_pts'/Computed(
        lambda ctx: ctx.npts if ctx.npts > 0 else ctx._.Header.header.npts),
    'exponent'/Computed(
        lambda ctx: ctx.exp if 0 < ctx.exp < 128 else ctx._.Header.header.exp),
    'raw_x'/If(this._.Header.TFlags.use_subfile_xs,
               LazyField(Array(this.num_pts, Int32sl))),
    'raw_y'/IfThenElse(this.float_y,
                       LazyField(Array(this.num_pts, Float32l)),
                       LazyField(Array(this.num_pts, Int32sl)))
)

LogData = Struct(
    'log_start'/Tell,
    'sizd'/Int32sl,
    'sizm'/Int32sl,
    'text_offset'/Int32sl,
    'bins'/Int32sl,
    'dsks'/Int32sl,
    Padding(44),
    'content'/Pointer(this.log_start + this.text_offset,
                      LazyField(Bytes(this.sizd)))
)

# The entire file.
SPCFile = Struct(
    'Header'/Header,
    'xvals'/If(this.Header.TFlags.has_xs,
               LazyField(Array(this.Header.header.npts, Float32l))),
    'Subfile'/Array(this.Header.header.nsub, Subfile),
    'LogData'/If(this.Header.header.log_offset != 0,
                 Pointer(this.Header.header.log_offset, LogData))
)


def prettyprint(data):
  np.set_printoptions(precision=4, suppress=True)
  version = data.Header.version
  h = data.Header.header
  print('SPC file, version %s (%s)' % (version, VERSIONS[version]))
  d = h.date
  try:
    print('Date:', datetime(d.year, d.month, d.day, d.hour, d.minute))
  except ValueError:
    pass  # Sometimes dates are all zeros
  if hasattr(h, 'experiment_type'):
    print('Experiment:', EXPERIMENT_TYPES[h.experiment_type])
  print('X axis:', X_AXIS_LABELS[h.xtype])
  print('Y axis:', Y_AXIS_LABELS[h.ytype])
  if hasattr(h, 'ztype'):
    print('Z axis:', X_AXIS_LABELS[h.ztype])
  if hasattr(h, 'wtype'):
    print('W axis:', X_AXIS_LABELS[h.wtype])
  if data.xvals is not None:
    assert h.TFlags.has_xs
    x = np.array(data.xvals())
    print('X (%d values):' % len(x), x)
  else:
    print('X: linspace(%g, %g, %d)' % (h.first, h.last, h.npts))
  print('%d subfiles' % len(data.Subfile))
  for i, sub in enumerate(data.Subfile, start=1):
    print('Subfile %d:' % i, sub)
  if data.LogData is not None:
    print('LogData:')
    try:
      print(data.LogData.content())
    except ConstructError as e:
      print('    Error reading log:', e)


def _convert_arrays(data):
  '''Generates a sequence of properly-converted (x,y) array pairs,
  one for each subpage.'''
  h = data.Header.header
  if data.xvals is not None:
    assert h.TFlags.has_xs
    x_vals = np.array(data.xvals())
  else:
    x_vals = np.linspace(h.first, h.last, int(h.npts))
  for sub in data.Subfile:
    if sub.raw_x is None:
      x = x_vals
    else:
      x = np.array(sub.raw_x(), dtype=float) * 2**(sub.exponent-32)

    if sub.float_y:
      # If it's floating point y data, we're done.
      y = np.array(sub.raw_y(), dtype=float)
    else:
      yraw = np.array(sub.raw_y(), dtype=np.int32)
      if data.Header.version == b'M':
        # old version needs different raw_y handling
        # TODO: fix this mess, if possible
        yraw = yraw.view(np.uint16).byteswap().view(np.uint32).byteswap()
      y = yraw * 2**(sub.exponent-32)
    yield x, y


def plot(data):
  from matplotlib import pyplot
  h = data.Header.header
  x_label = X_AXIS_LABELS[h.xtype]
  y_label = Y_AXIS_LABELS[h.ytype]
  for x, y in _convert_arrays(data):
    pyplot.figure()
    pyplot.plot(x, y)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
  pyplot.show()


def parse_traj(fh):
  # Parser requires binary file mode
  if hasattr(fh, 'mode') and 'b' not in fh.mode:
    fh = open(fh.name, 'rb')
  data = SPCFile.parse_stream(fh)
  h = data.Header.header
  assert h.nsub == 1, 'parse_traj only supports 1 SPC subfile'
  for x, y in _convert_arrays(data):
    return np.transpose((x, y))


if __name__ == '__main__':
  from optparse import OptionParser
  op = OptionParser()
  op.add_option('--print', action='store_true', dest='_print')
  op.add_option('--plot', action='store_true')
  opts, args = op.parse_args()
  if len(args) != 1:
    op.error('Supply exactly one filename argument.')
  if not (opts._print or opts.plot):
    op.error('Must supply either --plot or --print')
  data = SPCFile.parse_stream(open(args[0], 'rb'))
  if opts._print:
    prettyprint(data)
  if opts.plot:
    plot(data)
