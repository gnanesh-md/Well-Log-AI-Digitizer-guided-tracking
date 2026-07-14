import numpy as np
import cv2
from guided_curve_tracker import trace_guided_curve
import guided_curve_tracker

# Monkey patch _fast_dp_full to remove traverse_cost
orig_fast_dp = guided_curve_tracker._fast_dp_full
def fake_fast_dp(sub, dxs, move_pen, curvature_penalty, start_x, start_cost, endpoint_slope_weight, start_slope, vsub, hsub, ride_penalty, max_slope_px):
    # just run the python version with traverse_cost = 0
    pass

# We will just patch the Python DP since the Numba one is hard to monkey patch.
guided_curve_tracker._HAS_NUMBA = False

orig_trace_dp = guided_curve_tracker._trace_segment_dp
def patched_trace_dp(*args, **kwargs):
    # We will temporarily modify the source code to test this.
    pass

