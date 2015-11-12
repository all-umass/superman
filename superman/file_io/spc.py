import numpy as np
from datetime import datetime

from construct import (
    Anchor, Array, BitStruct, Byte, Embed, FieldError, Flag, If, IfThenElse,
    LFloat32, LFloat64, OnDemand, Padding, Pointer, SLInt16, SLInt32, String,
    Struct, Switch, ULInt32, Value
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
    Flag('has_xs'),
    Flag('use_subfile_xs'),
    Flag('use_catxt_labels'),
    Flag('ordered_subtimes'),
    Flag('arbitrary_time'),
    Flag('multiple_ys'),
    Flag('enable_experiment'),
    Flag('short_y')
)

DateVersionK = BitSplitter(
    ULInt32('Date'),
    minute=(0, 6),
    hour=(6, 5),
    day=(11, 5),
    month=(16, 4),
    year=(20, 12)
)

DateVersionM = Struct(
    'Date',
    SLInt16('year1900'),
    # XXX: guessing that we count years since 1900
    Value('year', lambda ctx: ctx.year1900 + 1900),
    Byte('month'),
    Byte('day'),
    Byte('hour'),
    Byte('minute')
)

HeaderVersionK = Struct(
    '',
    Byte('experiment_type'),
    Byte('exp'),
    SLInt32('npts'),
    LFloat64('first'),
    LFloat64('last'),
    SLInt32('nsub'),
    Byte('xtype'),
    Byte('ytype'),
    Byte('ztype'),
    Byte('post'),
    DateVersionK,
    Padding(9),
    FixedSizeCString('source', 9),
    SLInt16('peakpt'),
    Padding(32),
    FixedSizeCString('comment', 130),
    FixedSizeCString('catxt', 30),
    SLInt32('log_offset'),
    SLInt32('mods'),
    Byte('procs'),
    Byte('level'),
    SLInt16('sampin'),
    LFloat32('factor'),
    FixedSizeCString('method', 48),
    LFloat32('zinc'),
    SLInt32('wplanes'),
    LFloat32('winc'),
    Byte('wtype'),
    Padding(187)
)

HeaderVersionM = Struct(
    '',
    SLInt16('exp'),
    LFloat32('npts'),
    LFloat32('first'),
    LFloat32('last'),
    Byte('xtype'),
    Byte('ytype'),
    DateVersionM,
    Padding(8),  # res
    SLInt16('peakpt'),
    SLInt16('nscans'),
    Padding(28),  # spare
    FixedSizeCString('comment', 130),
    FixedSizeCString('catxt', 30),
    # only one subfile supported by this version
    Value('nsub', lambda ctx: 1),
    # log data is not supported by this version
    Value('log_offset', lambda ctx: 0)
)


def _wrong_version_error(ctx):
  raise NotImplementedError('SPC version %s is not implemented' % ctx.version)

Header = Struct(
    'Header',
    TFlags,
    String('version', 1),
    Switch('', lambda ctx: ctx.version, {
        'K': Embed(HeaderVersionK),
        'L': Value('', _wrong_version_error),
        'M': Embed(HeaderVersionM),
    }, default=Value('', _wrong_version_error))
)

Subfile = Struct(
    'Subfile',
    Byte('flags'),
    Byte('exp'),
    SLInt16('index'),
    LFloat32('time'),
    LFloat32('next'),
    LFloat32('nois'),
    SLInt32('npts'),
    SLInt32('scan'),
    LFloat32('wlevel'),
    Padding(4),
    Value('float_y', lambda ctx: ctx.exp == 128),
    Value('num_pts',
          lambda ctx: ctx.npts if ctx.npts > 0 else ctx._.Header.npts),
    Value('exponent',
          lambda ctx: ctx.exp if 0 < ctx.exp < 128 else ctx._.Header.exp),
    If(lambda ctx: ctx._.Header.TFlags.use_subfile_xs,
       OnDemand(Array(lambda ctx: ctx.num_pts, SLInt32('raw_x')))),
    IfThenElse('raw_y',
               lambda ctx: ctx.float_y,
               OnDemand(Array(lambda ctx: ctx.num_pts, LFloat32(''))),
               OnDemand(Array(lambda ctx: ctx.num_pts, SLInt32(''))))
)

LogData = Struct(
    'LogData',
    Anchor('log_start'),
    SLInt32('sizd'),
    SLInt32('sizm'),
    SLInt32('text_offset'),
    SLInt32('bins'),
    SLInt32('dsks'),
    Padding(44),
    Pointer(lambda ctx: ctx.log_start + ctx.text_offset,
            OnDemand(String('content', lambda ctx: ctx.sizd)))
)

# The entire file.
SPCFile = Struct(
    'SPCFile',
    Header,
    If(lambda ctx: ctx.Header.TFlags.has_xs,
       OnDemand(Array(lambda ctx: ctx.Header.npts, LFloat32('xvals')))),
    Array(lambda ctx: ctx.Header.nsub, Subfile),
    If(lambda ctx: ctx.Header.log_offset != 0,
       Pointer(lambda ctx: ctx.Header.log_offset, LogData))
)


def prettyprint(data):
  np.set_printoptions(precision=4, suppress=True)
  h = data.Header
  d = h.Date
  print 'SPC file, version %s (%s)' % (h.version, VERSIONS[h.version])
  try:
    print 'Date:', datetime(d.year, d.month, d.day, d.hour, d.minute)
  except ValueError:
    pass  # Sometimes dates are not provided, and are all zeros
  if hasattr(h, 'experiment_type'):
    print 'Experiment:', EXPERIMENT_TYPES[h.experiment_type]
  print 'X axis:', X_AXIS_LABELS[h.xtype]
  print 'Y axis:', Y_AXIS_LABELS[h.ytype]
  if hasattr(h, 'ztype'):
    print 'Z axis:', X_AXIS_LABELS[h.ztype]
  if hasattr(h, 'wtype'):
    print 'W axis:', X_AXIS_LABELS[h.wtype]
  if data.xvals is not None:
    assert h.TFlags.has_xs
    print 'X:', np.array(data.xvals.value)
  else:
    print 'X: linspace(%g, %g, %d)' % (h.first, h.last, h.npts)
  print '%d subfiles' % len(data.Subfile)
  for i, sub in enumerate(data.Subfile):
    print 'Subfile %d:' % (i+1), sub
  if data.LogData is not None:
    print 'LogData:'
    try:
      print data.LogData.content.value
    except FieldError as e:
      print '    Error reading log:', e


def _convert_arrays(data):
  '''Generates a sequence of properly-converted (x,y) array pairs,
  one for each subpage.'''
  h = data.Header
  if data.xvals is not None:
    assert h.TFlags.has_xs
    x_vals = np.array(data.xvals.value)
  else:
    x_vals = np.linspace(h.first, h.last, int(h.npts))
  for sub in data.Subfile:
    if sub.raw_x is None:
      x = x_vals
    else:
      x = np.array(sub.raw_x.value, dtype=float) * 2**(sub.exponent-32)

    if sub.float_y:
      # If it's floating point y data, we're done.
      y = np.array(sub.raw_y.value, dtype=float)
    else:
      yraw = np.array(sub.raw_y.value, dtype=np.int32)
      if h.version == 'M':
        # old version needs different raw_y handling
        # TODO: fix this mess, if possible
        yraw = yraw.view(np.uint16).byteswap().view(np.uint32).byteswap()
      y = yraw * 2**(sub.exponent-32)
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
