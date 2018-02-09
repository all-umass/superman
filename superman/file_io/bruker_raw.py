'''
Siemens/Bruker Diffrac-AT Raw Format
 * https://github.com/wojdyr/xylib/blob/master/xylib/brucker_raw.cpp
'''
from __future__ import absolute_import
import numpy as np
from construct import (
  Padding, Struct, Switch, Bytes, Array, Const, Computed,
  Int32ul, Int16ul, Float64l, Float32l, this, Check
)
from .construct_utils import FixedSizeCString, LazyField

Block_v2 = Struct(
    'header_len'/Int16ul,
    'num_steps'/Int16ul,
    Padding(4),
    'time_per_step'/Float32l,
    'x_step'/Float64l,
    'x_start'/Float64l,
    Padding(26),
    'temperature'/Int16ul,
    Padding(this.header_len - 48),
    'y'/LazyField(Array(this.num_steps, Float32l))
)

RAW_v2 = Struct(
    'num_blocks'/Int32ul,
    Padding(162),
    'date_time_measure'/FixedSizeCString(20),
    'anode_material'/FixedSizeCString(2),
    'lambda1'/Float32l,
    'lambda2'/Float32l,
    'intensity_ratio'/Float32l,
    Padding(8),
    'sample_runtime'/Float32l,
    Padding(42),
    'blocks'/Array(this.num_blocks, Block_v2)
)

Block_v101 = Struct(
    'header_len'/Int32ul,
    Check(this.header_len == 304),
    'num_steps'/Int32ul,
    'start_theta'/Float64l,
    'start_2theta'/Float64l,
    Padding(76),
    'high_voltage'/Float32l,
    'amplifier_gain'/Float32l,
    'discriminator_1_lower_level'/Float32l,
    Padding(64),
    'step_size'/Float64l,
    Padding(8),
    'time_per_step'/Float32l,
    Padding(12),
    'rpm'/Float32l,
    Padding(12),
    'generator_voltage'/Int32ul,
    'generator_current'/Int32ul,
    Padding(8),
    'used_lambda'/Float64l,
    Padding(8),
    'supplementary_headers_size'/Int32ul,
    Padding(this.supplementary_headers_size + 44),
    'y'/LazyField(Array(this.num_steps, Float32l))
)

RAW_v101 = Struct(
    Const(b'.01'),
    Padding(1),
    # 1 -> done, 2 -> active, 3 -> aborted, 4 -> interrupted
    'file_status'/Int32ul,
    'num_blocks'/Int32ul,
    'measure_date'/FixedSizeCString(10),
    'measure_time'/FixedSizeCString(10),
    'user'/FixedSizeCString(72),
    'site'/FixedSizeCString(218),
    'sample_id'/FixedSizeCString(60),
    'comment'/FixedSizeCString(160),
    Padding(62),
    'anode_material'/FixedSizeCString(4),
    Padding(4),
    'alpha_average'/Float64l,
    'alpha1'/Float64l,
    'alpha2'/Float64l,
    'beta'/Float64l,
    'alpha_ratio'/Float64l,
    Padding(8),
    'measurement_time'/Float32l,
    Padding(44),
    'blocks'/Array(this.num_blocks, Block_v101)
)


def _unsupported_version(ctx):
  v = b'1' if ctx.version == b' ' else ctx.version
  raise NotImplementedError('Bruker RAW version %r is not implemented' % v)

RAW = Struct(
  Const(b'RAW'),
  'version'/Bytes(1),
  'body'/Switch(this.version, {
    b'2': RAW_v2,
    b'1': RAW_v101,
  }, default=Computed(_unsupported_version))
)


def parse_raw(fh):
  # Parser requires binary file mode
  if hasattr(fh, 'mode') and 'b' not in fh.mode:
    fh = open(fh.name, 'rb')
  data = RAW.parse_stream(fh)
  assert data.body.num_blocks == 1, 'Bruker RAW files w/ only 1 block supported'
  block, = data.body.blocks
  n = block.num_steps
  if data.version == b'1':
    x_start = block.start_2theta
    x_stop = x_start + n * block.step_size
  elif data.version == b'2':
    x_start = block.x_start
    x_stop = x_start + n * block.x_step
  x = np.linspace(x_start, x_stop, num=n, endpoint=False)
  y = np.array(block.y(), dtype=np.float32)
  return np.column_stack((x, y))
