from construct import Adapter, Container, IfThenElse, Pass, Bytes

try:
  from construct import LazyField
except ImportError:
  # pre version 2.9 name
  from construct import OnDemand as LazyField


class BitSplitter(Adapter):
  '''Hacks around lack of little-endian BitField support.
  Taken from the internet: http://construct.wikispaces.com/bitfields'''
  def __init__(self, subcon, **fields):
    Adapter.__init__(self, subcon)
    self.fields = fields

  def _encode(self, obj, ctx):
    num = 0
    for name, (offset, size) in self.fields.items():
      val = getattr(obj, name) & (2**size-1)
      num |= val << offset
    return num

  def _decode(self, obj, ctx):
    c = Container()
    for name, (offset, size) in self.fields.items():
      setattr(c, name, (obj >> offset) & (2**size-1))
    return c


class FixedSizeCString(Adapter):
  '''Marries a C-style null-terminated string with a fixed-length field.'''
  def __init__(self, size_fn):
    Adapter.__init__(self, Bytes(size_fn))

  def _decode(self, obj, ctx):
    return obj.split(b'\0',1)[0]

  def _encode(self, obj, ctx):
    size = self._sizeof(ctx, None)
    return obj.ljust(size, b'\0')[:size]


def FunctionSwitch(cond_pairs, default=Pass):
  '''Function-based switch statement.
  Evaluates (test, subcon) pairs in order,
  taking whichever test() evaluates true first.'''
  res = default
  for test_fn, subcon in reversed(cond_pairs):
    res = IfThenElse(test_fn, subcon, res)
  return res
