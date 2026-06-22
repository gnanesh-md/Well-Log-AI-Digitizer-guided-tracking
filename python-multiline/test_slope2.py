import cv2, numpy as np, time
from guided_curve_tracker import trace_guided_curve

img = np.full((6000, 1000, 3), 255, dtype=np.uint8)
ys = np.arange(6000)
xa = (500 + 120*np.sin(ys/90.0)).astype(int)
for y in ys[:-1]:
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)

anchors = [(xa[100], 100), (xa[5900], 5900)]

t0 = time.time()
res8 = trace_guided_curve(img, anchors, curve_style="solid", max_slope_px=8)
t1 = time.time()

pts8 = np.array(res8["points"])
err8 = np.mean(np.abs(pts8[:, 0] - xa[pts8[:, 1]]))

print(f"max_slope=8: {t1-t0:.3f}s, error={err8:.3f}")
