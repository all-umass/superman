# Import class wrappers for each type of baseline alg.
from airpls import AirPLS
from als import ALS
from dietrich import Dietrich
from fabc import FABC
from kajfosz_kwiatek import KajfoszKwiatek
from mario import Mario
from median import MedianFilter
from mpls import MPLS
# from ob import OB
from polyfit import PolyFit
from rubberband import Rubberband
from wavelet import Wavelet

BL_CLASSES = dict(
    airpls=AirPLS, als=ALS, dietrich=Dietrich, fabc=FABC, kk=KajfoszKwiatek,
    mario=Mario, median=MedianFilter, mpls=MPLS, polyfit=PolyFit,
    rubberband=Rubberband, wavelet=Wavelet
)
