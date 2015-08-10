# Import class wrappers for each type of baseline alg.
from airpls import AirPLS
from als import ALS
from dietrich import Dietrich
from fabc import FABC
from kajfosz_kwiatek import KajfoszKwiatek
from mario import Mario
# from ob import OB
from wavelet import Wavelet
from median import MedianFilter
from polyfit import PolyFit
from rubberband import Rubberband

BL_CLASSES = dict(als=ALS, mario=Mario, dietrich=Dietrich, fabc=FABC,
                  airpls=AirPLS, kk=KajfoszKwiatek, wavelet=Wavelet,
                  median=MedianFilter, polyfit=PolyFit, rubberband=Rubberband)

