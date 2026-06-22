import cv2, numpy as np, time
import cProfile, pstats
from guided_curve_tracker import trace_guided_curve

img = np.full((6000, 1000, 3), 255, dtype=np.uint8)
ys = np.arange(6000)
xa = (500 + 120*np.sin(ys/90.0)).astype(int)
for y in ys[:-1]:
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)

anchors = [(xa[100], 100), (xa[5900], 5900)]

def run_trace():
    trace_guided_curve(img, anchors, curve_style="solid")

print("Starting profile...")
profiler = cProfile.Profile()
profiler.enable()
run_trace()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats(15)
