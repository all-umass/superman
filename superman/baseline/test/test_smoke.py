from __future__ import absolute_import
import numpy as np
import unittest
from six import with_metaclass

from .. import BL_CLASSES

# static test fixture data
y = np.linspace(0, 1, 100)  # linear baseline
y += 1e-4 / (y-0.25)**2 + 5e-4 / ((y-0.75)**2 + 1e-4)  # 2 peaks
bands = np.arange(len(y))
bands[45:] += 3  # make two segments
yyy = np.ascontiguousarray(np.row_stack((y, np.sqrt(y/y.max()), 1.1 * y)))


class SmokeTestMeta(type):
  def __new__(mcs, name, bases, ns):
    def gen_param_ranges_test(blr_cls):
      def _test(self):
        pr = blr_cls().param_ranges()
        for key in pr:
          min_, max_, scale = pr[key]
          self.assertLess(min_, max_)
          self.assertIn(scale, ('linear','log','integer'))
      return _test

    def gen_fit_test(blr_cls, ints, segmented):
      def _test(self):
        bl = blr_cls()
        bl.fit(bands, ints, segment=segmented)
        self.assertEqual(bl.baseline.shape, ints.shape)
      return _test

    for blr_key, blr_cls in BL_CLASSES.items():
      ns['test_param_ranges_' + blr_key] = gen_param_ranges_test(blr_cls)
      ns['test_fit_1d_' + blr_key] = gen_fit_test(blr_cls, y, False)
      ns['test_fit_2d_' + blr_key] = gen_fit_test(blr_cls, yyy, False)
      ns['test_fit_1d_segmented_' + blr_key] = gen_fit_test(blr_cls, y, True)
      ns['test_fit_2d_segmented_' + blr_key] = gen_fit_test(blr_cls, yyy, True)

    return type.__new__(mcs, name, bases, ns)


class SmokeTest(with_metaclass(SmokeTestMeta, unittest.TestCase)):
  pass


if __name__ == '__main__':
  unittest.main()
