'''
Siemens/Bruker Diffrac-AT Raw Format
 * https://github.com/wojdyr/xylib/blob/master/xylib/brucker_raw.cpp
'''
import numpy as np
from construct import (
  Padding, Struct, ULInt32, ULInt16, Switch, String, LFloat64, LFloat32, Const,
  Array, Magic, OnDemand, Rename, Embed, Value
)
from construct_utils import FixedSizeCString

Block_v2 = Struct(
    'RAW v2 data block',
    ULInt16('header_len'),
    ULInt16('num_steps'),
    Padding(4),
    LFloat32('time_per_step'),
    LFloat64('x_step'),
    LFloat64('x_start'),
    Padding(26),
    ULInt16('temperature'),
    Padding(lambda ctx: ctx.header_len - 48),
    OnDemand(Array(lambda ctx: ctx.num_steps, LFloat32('y')))
)

RAW_v2 = Struct(
    'RAW v2',
    ULInt32('num_steps'),
    Padding(162),
    FixedSizeCString('date_time_measure', 20),
    FixedSizeCString('anode_material', 2),
    LFloat32('lambda1'),
    LFloat32('lambda2'),
    LFloat32('intensity_ratio'),
    Padding(8),
    LFloat32('sample_runtime'),
    Padding(42),
    Rename('blocks', Array(lambda ctx: ctx.num_steps, Block_v2))
)

Block_v101 = Struct(
    'RAW v1.01 data block',
    Const(ULInt32('header_len'), 304),
    ULInt32('num_steps'),
    LFloat64('start_theta'),
    LFloat64('start_2theta'),
    Padding(76),
    LFloat32('high_voltage'),
    LFloat32('amplifier_gain'),
    LFloat32('discriminator_1_lower_level'),
    Padding(64),
    LFloat64('step_size'),
    Padding(8),
    LFloat32('time_per_step'),
    Padding(12),
    LFloat32('rpm'),
    Padding(12),
    ULInt32('generator_voltage'),
    ULInt32('generator_current'),
    Padding(8),
    LFloat64('used_lambda'),
    Padding(8),
    ULInt32('supplementary_headers_size'),
    Padding(lambda ctx: ctx.supplementary_headers_size + 44),
    OnDemand(Array(lambda ctx: ctx.num_steps, LFloat32('y')))
)

RAW_v101 = Struct(
    'RAW v1.01',
    Magic('.01'),
    Padding(1),
    # 1 -> done, 2 -> active, 3 -> aborted, 4 -> interrupted
    ULInt32('file_status'),
    ULInt32('num_blocks'),
    FixedSizeCString('measure_date', 10),
    FixedSizeCString('measure_time', 10),
    FixedSizeCString('user', 72),
    FixedSizeCString('site', 218),
    FixedSizeCString('sample_id', 60),
    FixedSizeCString('comment', 160),
    Padding(62),
    FixedSizeCString('anode_material', 4),
    Padding(4),
    LFloat64('alpha_average'),
    LFloat64('alpha1'),
    LFloat64('alpha2'),
    LFloat64('beta'),
    LFloat64('alpha_ratio'),
    Padding(8),
    LFloat32('measurement_time'),
    Padding(44),
    Rename('blocks', Array(lambda ctx: ctx.num_blocks, Block_v101))
)


def _unsupported_version(ctx):
  v = '1' if ctx.version == ' ' else ctx.version
  raise NotImplementedError('Bruker RAW version %r is not implemented' % v)

RAW = Struct(
  'Bruker RAW file',
  Magic('RAW'),
  String('version', 1),
  Switch('', lambda ctx: ctx.version, {
    ' ': Value('', _unsupported_version),
    '2': Embed(RAW_v2),
    '1': Embed(RAW_v101)
  }, default=Value('', _unsupported_version))
)


def parse_raw(fh):
  # Parser requires binary file mode
  if hasattr(fh, 'mode') and 'b' not in fh.mode:
    fh = open(fh.name, 'rb')
  data = RAW.parse_stream(fh)
  assert data.num_blocks == 1, 'Bruker RAW files w/ only 1 block supported'
  block, = data.blocks
  n = block.num_steps
  if data.version == '1':
    x_start = block.start_2theta
    x_stop = x_start + n * block.step_size
  elif data.version == '2':
    x_start = block.x_start
    x_stop = x_start + n * block.x_step
  x = np.linspace(x_start, x_stop, num=n, endpoint=False)
  y = np.array(block.y.value, dtype=np.float32)
  return np.column_stack((x, y))
