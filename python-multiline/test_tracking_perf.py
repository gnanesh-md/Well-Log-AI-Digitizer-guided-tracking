import time
import numpy as np
import cv2
from guided_curve_tracker import trace_guided_curve
import guided_curve_tracker

# Create synthetic thin spiky curve
H, W = 1000, 500
img = np.full((H, W, 3), (230, 230, 230), dtype=np.uint8)
# Add gridlines
for i in range(0, H, 50):
    cv2.line(img, (0, i), (W, i), (200, 200, 200), 1)
for i in range(0, W, 50):
    cv2.line(img, (i, 0), (i, H), (200, 200, 200), 1)

# Draw thin spike
pts = []
for y in range(0, 300): pts.append((400, y))
for y in range(300, 400):
    # spike to 100
    x = 400 - int((y - 300) / 100.0 * 300)
    pts.append((x, y))
for y in range(400, 500):
    x = 100 + int((y - 400) / 100.0 * 300)
    pts.append((x, y))
for y in range(500, 1000): pts.append((400, y))

for i in range(len(pts)-1):
    cv2.line(img, pts[i], pts[i+1], (100, 100, 100), 2)

anchors = [(400, 50), (400, 950)]
start = time.time()
res = trace_guided_curve(img, anchors, curve_style="solid")
dur = time.time() - start

print(f"Time: {dur*1000:.1f} ms")
min_x = min(p[0] for p in res["points"])
print(f"Min X: {min_x} (Expected ~100)")
print(f"HAS NUMBA: {guided_curve_tracker._HAS_NUMBA}")
if min_x > 150:
    print("FAILED")
    exit(1)
print("SUCCESS")
