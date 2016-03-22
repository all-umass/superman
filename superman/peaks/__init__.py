from __future__ import absolute_import
from .bump_fit import BumpFit
from .derivative import Derivative
from .savitzky_golay import SavitzkyGolay
from .threshold import Threshold
from .wavelet import Wavelet

PEAK_CLASSES = {
    'wavelet': Wavelet,
    'sg': SavitzkyGolay,
    'threshold': Threshold,
    'deriv': Derivative,
    'bumpfit': BumpFit,
}
