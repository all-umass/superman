import numpy as np
from collections import defaultdict
from datetime import datetime

from construct import (
    Flag, ULInt32, ULInt16, LFloat32, LFloat64, BitStruct, String, Byte, Struct,
    Padding, Value, If, OnDemand, IfThenElse, Array, Pointer, Anchor, Terminator
)
from construct_utils import BitSplitter, FixedSizeCString

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
    'K': 'new LSB 1st',
    'L': 'new MSB 1st',
    'M': 'old format'
}

TFlags = BitStruct(
    'TFlags',
    Flag('short_y'),
    Flag('enable_experiment'),
    Flag('multiple_ys'),
    Flag('arbitrary_time'),
    Flag('ordered_subtimes'),
    Flag('use_catxt_labels'),
    Flag('use_subfile_xs'),
    Flag('has_xs')
)

Date = BitSplitter(
    ULInt32('Date'),
    minute=(0, 6),
    hour=(6, 5),
    day=(11, 5),
    month=(16, 4),
    year=(20, 12)
)

Header = Struct(
    'Header',
    TFlags,
    String('version', 1),
    Byte('experiment_type'),
    Byte('exp'),
    ULInt32('npts'),
    LFloat64('first'),
    LFloat64('last'),
    ULInt32('nsub'),
    Byte('xtype'),
    Byte('ytype'),
    Byte('ztype'),
    Byte('post'),
    Date,
    Padding(9),
    FixedSizeCString('source', 9),
    ULInt16('peakpt'),
    Padding(32),
    FixedSizeCString('comment', 130),
    String('catxt', 30),
    ULInt32('log_offset'),
    ULInt32('mods'),
    Byte('procs'),
    Byte('level'),
    ULInt16('sampin'),
    LFloat32('factor'),
    FixedSizeCString('method', 48),
    LFloat32('zinc'),
    ULInt32('wplanes'),
    LFloat32('winc'),
    Byte('wtype'),
    Padding(187)
)

Subfile = Struct(
    'Subfile',
    Byte('flags'),
    Byte('exp'),
    ULInt16('index'),
    LFloat32('time'),
    LFloat32('next'),
    LFloat32('nois'),
    ULInt32('npts'),
    ULInt32('scan'),
    LFloat32('wlevel'),
    Padding(4),
    Value('float_y', lambda ctx: ctx.exp == 128),
    Value('num_pts',
          lambda ctx: ctx.npts if ctx.npts > 0 else ctx._.Header.npts),
    Value('exponent',
          lambda ctx: ctx.exp if 0 < ctx.exp < 128 else ctx._.Header.exp),
    If(lambda ctx: ctx._.Header.TFlags.use_subfile_xs,
       OnDemand(Array(lambda ctx: ctx.num_pts, ULInt32('raw_x')))),
    IfThenElse('raw_y',
               lambda ctx: ctx.float_y,
               OnDemand(Array(lambda ctx: ctx.num_pts, LFloat32(''))),
               OnDemand(Array(lambda ctx: ctx.num_pts, ULInt32(''))))
)

LogData = Struct(
    'LogData',
    Anchor('log_start'),
    ULInt32('sizd'),
    ULInt32('sizm'),
    ULInt32('text_offset'),
    ULInt32('bins'),
    ULInt32('dsks'),
    Padding(44),
    Pointer(lambda ctx: ctx.log_start + ctx.text_offset,
            String('content', lambda ctx: ctx.sizd))
)

# The entire file.
SPCFile = Struct(
    'SPCFile',
    Header,
    If(lambda ctx: ctx.Header.TFlags.has_xs,
       Array(lambda ctx: ctx.Header.npts, LFloat32('xvals'))),
    Array(lambda ctx: ctx.Header.nsub, Subfile),
    If(lambda ctx: ctx.Header.log_offset != 0,
       Pointer(lambda ctx: ctx.Header.log_offset, LogData)),
    Terminator
)


def prettyprint(data):
  np.set_printoptions(precision=4, suppress=True)
  h = data.Header
  d = h.Date
  print 'SPC file, version', h.version
  print 'Date:', datetime(d.year, d.month, d.day, d.hour, d.minute)
  print 'Experiment:', EXPERIMENT_TYPES[h.experiment_type]
  print 'X axis:', X_AXIS_LABELS[h.xtype]
  print 'Y axis:', Y_AXIS_LABELS[h.ytype]
  print 'Z axis:', X_AXIS_LABELS[h.ztype]
  print 'W axis:', X_AXIS_LABELS[h.wtype]
  if data.xvals is not None:
    assert not h.TFlags.has_xs
    print 'X:', np.array(data.xvals)
  else:
    print 'X: linspace(%g, %g, %d)' % (h.first, h.last, h.npts)
  for i, sub in enumerate(data.Subfile):
    print 'Subfile %d:' % (i+1), sub
  print 'LogData:', data.LogData


def _convert_arrays(data):
  '''Generates a sequence of properly-converted (x,y) array pairs,
  one for each subpage.'''
  h = data.Header
  if data.xvals is not None:
    assert not h.TFlags.has_xs
    x_vals = np.array(data.xvals)
  else:
    x_vals = np.linspace(h.first, h.last, h.npts)
  for sub in data.Subfile:
    if sub.raw_x is None:
      x = x_vals
    else:
      x = np.array(sub.raw_x.value) * 2**(sub.exponent-32)
    y = np.array(sub.raw_y.value)
    if not sub.float_y:
      y *= 2**(sub.exponent-32)
    yield x, y


def plot(data):
  from matplotlib import pyplot
  h = data.Header
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
  assert data.Header.nsub == 1, 'parse_traj only support 1 subfile for now'
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
    op.error('Must supply either --plot or --print.')
  data = SPCFile.parse_stream(open(args[0], 'rb'))
  if opts._print:
    prettyprint(data)
  if opts.plot:
    plot(data)
