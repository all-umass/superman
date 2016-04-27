from __future__ import absolute_import
import warnings

# Import class wrappers for each type of baseline alg.
from .airpls import AirPLS
from .als import ALS
from .dietrich import Dietrich
from .fabc import FABC
from .kajfosz_kwiatek import KajfoszKwiatek
from .median import MedianFilter
from .mpls import MPLS
# from .ob import OB
from .polyfit import PolyFit
from .rubberband import Rubberband
from .tophat import Tophat

BL_CLASSES = dict(
    airpls=AirPLS, als=ALS, dietrich=Dietrich, fabc=FABC, kk=KajfoszKwiatek,
    median=MedianFilter, mpls=MPLS, polyfit=PolyFit, rubberband=Rubberband,
    tophat=Tophat
)

# add baseline methods that might fail to import
try:
  from .mario import Mario
except ImportError:
  warnings.warn('Failed to import mario baseline: '
                'install cvxopt or scipy >= 0.15')
else:
  BL_CLASSES['mario'] = Mario

try:
  from .wavelet import Wavelet
except ImportError as e:
  warnings.warn('Failed to import wavelet baseline: %s' % e)
else:
  BL_CLASSES['wavelet'] = Wavelet
