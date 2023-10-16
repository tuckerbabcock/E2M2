__version__ = '0.0.1'

from .e2m2_driver import E2M2Driver
from .trmm_driver import TRMMDriver
from .trust_region import TrustRegion
from .calibration import AdditiveCalibration, MultiplicativeCalibration, calibrate
from .error_est import ErrorEstimate, PolynomialErrorPenalty
from .utils import estimate_lagrange_multipliers, get_active_constraints, constraint_violation, l1_merit_function, optimality