import cv2, numpy as np, time
from guided_curve_tracker import trace_guided_curve

img = np.full((6000, 1000, 3), 255, dtype=np.uint8)
ys = np.arange(6000)
xa = (500 + 120*np.sin(ys/90.0)).astype(int)
for y in ys[:-1]:
    cv2.line(img, (xa[y], y), (xa[y+1], y+1), (20, 20, 20), 2)

anchors = [(xa[100], 100), (xa[5900], 5900)]

t0 = time.time()
res15 = trace_guided_curve(img, anchors, curve_style="solid", max_slope_px=15)
t1 = time.time()

t2 = time.time()
res35 = trace_guided_curve(img, anchors, curve_style="solid", max_slope_px=35)
t3 = time.time()

pts15 = np.array(res15["points"])
pts35 = np.array(res35["points"])

err15 = np.mean(np.abs(pts15[:, 0] - xa[pts15[:, 1]]))
err35 = np.mean(np.abs(pts35[:, 0] - xa[pts35[:, 1]]))

print(f"max_slope=15: {t1-t0:.3f}s, error={err15:.3f}")
print(f"max_slope=35: {t3-t2:.3f}s, error={err35:.3f}")
